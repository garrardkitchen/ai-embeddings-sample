using OpenAI;
using Microsoft.Extensions.AI;
using System.ClientModel;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.VectorData;
using Microsoft.Extensions.Configuration.UserSecrets;
#pragma warning disable
using Microsoft.SemanticKernel.Connectors.InMemory;

public partial class OpenAISamples
{
    private static ConfigurationManager _configuration;
    private static readonly string _collectionName = "drupert-collection";

    /// <summary>
    /// Executes the Text Embedding process, which involves ingestion, semantic search, and generation.
    /// This process uses OpenAI's embedding and chat generation capabilities to create context-driven responses based on user input and external data.
    /// </summary>
    /// <returns>
    /// A <see cref="Task"/> that represents the asynchronous operation of the embedding process. Includes ingestion of data, querying, semantic search, and generation of contextually relevant outputs.
    /// </returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the necessary OpenAI configurations or credentials are missing.
    /// </exception>
    public static async Task TextEmbedding()
    {
        _configuration = new ConfigurationManager();
        _configuration.AddUserSecrets<OpenAISamples>();
        var openAiClient = CreateOpenAiClient();
        var chatClient = openAiClient.AsChatClient("gpt-4o-mini");
        var generator = openAiClient.GetEmbeddingClient("text-embedding-3-small").AsEmbeddingGenerator();
        var collection = await GetCollectionCreateIfNotExistsAsync();
        
        // user query
        var query = "I want a pleasant short story about a drupert. " +
                    "This story is to be less than 5 sentences. " +
                    "It has to make me want to laugh out loud. It must end with a hugging emoji " +
                    "and this emogi has to be on a new line. The story must have a title";

        string[] externalData = [
            "A Drupert is a fictional creature", 
            "A Sleepert is meant to make you fall asleep during school time", 
            "Chicken Jockie!",
            "A Drupert is meant to distract", 
            "A Drupert sole purpose is to make you laugh", 
            "If someone draws a Drupert during class time and you laugh, you'll likely be told off my your teacher"];
        
        // step 1: ingestion (chunking & indexing). Tokens will be used
        await IngestionAsync(collection, generator, externalData);
        // step 2: retrieval. Tokens will be used
        var searchResults = await SemanticSearchAsync(collection, generator, query);
        // step 3: augmentation
        var prompt = CreatePrompt(searchResults.Results, query);
        // step 4: generation. Tokens will be used
        Console.WriteLine(await chatClient.GetResponseAsync(new ChatMessage(ChatRole.User, prompt)));
    }

    /// <summary>
    /// Creates and initializes an instance of the <see cref="OpenAIClient"/> class with the necessary configurations and credentials.
    /// </summary>
    /// <returns>
    /// A configured instance of the <see cref="OpenAIClient"/> class used for interacting with OpenAI services.
    /// </returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the required configuration value for "GitHubModels:Token" is missing or null.
    /// </exception>
    private static OpenAIClient CreateOpenAiClient()
    {
        var openAIOptions = new OpenAIClientOptions()
        {
            Endpoint = new Uri("https://models.inference.ai.azure.com"),
        };
        var credential = new ApiKeyCredential(
            _configuration.GetValue<string>("GitHubModels:Token") 
            ?? throw new InvalidOperationException("Missing configuration: GitHubModels:Token. Ensure it is set in UserSecrets."));
        var model = new OpenAIClient(credential, openAIOptions);
        return model;
    }

    /// <summary>
    /// Generates a prompt for a story based on the provided semantic search results and user query.
    /// The generated prompt includes extracted facts presented in a structured format and incorporates user input for context.
    /// </summary>
    /// <param name="results">
    /// An asynchronous enumerable containing semantic search results represented as <see cref="VectorSearchResult{T}"/>
    /// objects. Each result includes an associated fact or record from the vector store.
    /// </param>
    /// <param name="userQuery">
    /// The user-provided query or input to guide the generation of the story prompt.
    /// </param>
    /// <returns>
    /// A formatted string representing the generated prompt for creating a story, which includes the extracted facts and
    /// the user's query. The prompt ensures information is cited and structured using fact extraction IDs.
    /// </returns>
    private static string CreatePrompt(IAsyncEnumerable<VectorSearchResult<VectorRecord>> results, string userQuery)
    {
        var contents = $$"""
                         Give an answer using ONLY information from the following product manual extracts.
                         If the product manual doesn't contain the information, you should say so. 
                         Do not make up information beyond what is given.
                         Whenever relevant, specify fact_extract id to cite the factual extract that your answer is based on. 
                         Please include the fact_extract ids that were used to generate this story, at the end of the story.
                         The format of this fact_extract ids must be, and they must be listed in ascending order:

                            Facts: [ids]

                         These are the facts about druperts:
                         {{string.Join(Environment.NewLine, results.ToBlockingEnumerable().Select(c => $"<fact_extract id='{c.Record.id}'>{c.Record.value}</fact_extract>"))}}

                         User question: {{{userQuery}}}
                         """;
        Console.WriteLine();
        Console.WriteLine(contents);
        Console.WriteLine();
        return contents;
    }

    /// <summary>
    /// Performs a semantic search on the specified vector store collection using the provided embedding generator and user query.
    /// </summary>
    /// <param name="collection">
    /// The collection representing the vector store where the search will be performed.
    /// </param>
    /// <param name="generator">
    /// The embedding generator used to compute the query embedding from the provided user query.
    /// </param>
    /// <param name="userQuery">
    /// The user-defined query string that will be used to generate embeddings for the semantic search.
    /// </param>
    /// <param name="productId">
    /// An optional product identifier used as a filter in the search results. Only records matching this identifier will be considered.
    /// </param>
    /// <returns>
    /// A task that represents the asynchronous operation. The task result contains the search results as a <see cref="VectorSearchResults{VectorRecord}"/>.
    /// </returns>
    private static async Task<VectorSearchResults<VectorRecord>> SemanticSearchAsync(
        IVectorStoreRecordCollection<int, VectorRecord> collection,
        IEmbeddingGenerator<string, Embedding<float>> generator,
        string userQuery,
        int? productId = null
        )
    {
        var queryEmbedding = await generator.GenerateEmbeddingVectorAsync(userQuery);

        var filter =
            new VectorSearchFilter()
                .EqualTo(nameof(VectorRecord.productId), productId);
        
        var searchOptions = new VectorSearchOptions
        {
            Top = 5,
            Filter = filter,
            IncludeVectors = true
        };
        
        // perform semantic search on the vector store
        var searchResults = await collection.VectorizedSearchAsync(queryEmbedding, searchOptions);
        return searchResults;
    }

    /// <summary>
    /// Processes the ingestion of an array of text prompts by generating embeddings using the given embedding generator
    /// and storing them in the specified vector store collection.
    /// </summary>
    /// <param name="collection">The vector store collection where the generated embeddings will be stored.</param>
    /// <param name="generator">The embedding generator used to create embeddings for the provided prompts.</param>
    /// <param name="prompts">An array of text prompts to be converted into embeddings and ingested into the collection.</param>
    /// <returns>
    /// A task that represents the asynchronous ingestion operation.
    /// </returns>
    private static async Task IngestionAsync(
        IVectorStoreRecordCollection<int, VectorRecord> collection,
        IEmbeddingGenerator<string, Embedding<float>> generator, 
        string[] prompts)
    {
        // Add prompts to the vector store
        var index = 0;
        foreach (var prompt in prompts)
        {
            Console.WriteLine($"{index}: {prompt}");
            var embedding = await generator.GenerateEmbeddingVectorAsync(prompt);
            var record = new VectorRecord(index++, prompt, embedding);
            collection.UpsertAsync(record);
        }
    }

    /// <summary>
    /// Retrieves a vector store collection by the specified name and creates it if it does not already exist.
    /// </summary>
    /// <returns>
    /// An instance of <see cref="IVectorStoreRecordCollection{TKey, TValue}"/> representing the vector store collection, which can be used for storing and querying vector records.
    /// </returns>
    private static async Task<IVectorStoreRecordCollection<int, VectorRecord>> GetCollectionCreateIfNotExistsAsync()
    {
        var vectorStore = new InMemoryVectorStore();
        await vectorStore.GetCollection<int, VectorRecord>(_collectionName, GetRecordDefinition()).CreateCollectionIfNotExistsAsync();
        IVectorStoreRecordCollection<int, VectorRecord> collection = vectorStore.GetCollection<int, VectorRecord>(_collectionName, GetRecordDefinition());

        return vectorStore.GetCollection<int, VectorRecord>(_collectionName, GetRecordDefinition());
    }

    /// <summary>
    /// Defines the structure and metadata used for creating and interacting with a vector store record collection.
    /// </summary>
    /// <returns>
    /// An instance of <see cref="VectorStoreRecordDefinition"/> that specifies properties, key, vector,
    /// and other configurations required by the vector store.
    /// </returns>
    private static VectorStoreRecordDefinition GetRecordDefinition()
    {
        return new VectorStoreRecordDefinition
        {
            Properties = new List<VectorStoreRecordProperty>
            {
                new VectorStoreRecordKeyProperty(nameof(VectorRecord.id), typeof(int)),
                new VectorStoreRecordDataProperty(nameof(VectorRecord.productId), typeof(int)) { IsFilterable = true },
                new VectorStoreRecordVectorProperty(nameof(VectorRecord.vector), typeof(ReadOnlyMemory<float>)) { Dimensions = 1536, DistanceFunction = DistanceFunction.CosineDistance },
                new VectorStoreRecordDataProperty(nameof(VectorRecord.value), typeof(string)) { IsFilterable = true },
            }
        };
    }
}

/// <summary>
/// Represents a record used in vector-based searches, containing the necessary properties for
/// semantic embeddings and metadata for filtering and retrieval.
/// </summary>
/// <param name="id">
/// The unique identifier for the vector record.
/// </param>
/// <param name="value">
/// The textual or other content associated with this record, typically the data being indexed for retrieval.
/// </param>
/// <param name="vector">
/// The vector representation (embedding) of the record, used for similarity-based searches in the vector store.
/// </param>
/// <param name="productId">
/// An optional product identifier to associate the record with a specific product, which can also be used as a filter during search.
/// </param>
record VectorRecord(int id, string value, ReadOnlyMemory<float> vector, int? productId = null);

