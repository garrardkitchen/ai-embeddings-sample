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
    private static string _collectionName = "druppert-collection";
    
    public static async Task TextEmbedding()
    {
        _configuration = new ConfigurationManager();
        _configuration.AddUserSecrets<OpenAISamples>(); // Add UserSecrets
        var openAiClient = CreateOpenAiClient();
        var chatClient = 
            openAiClient.AsChatClient("gpt-4o-mini");
        var generator =
            openAiClient.GetEmbeddingClient("text-embedding-3-small").AsEmbeddingGenerator();
        var store = await GetVectorStoreIfNotExistsThenCreateAsync();
        
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
        await IngestionAsync(store, generator, externalData);
        // step 2: retrieval. Tokens will be used
        var searchResults = await SemanticSearchAsync(store, generator, query);
        // step 3: augmentation
        var prompt = CreatePrompt(searchResults.Results, query);
        // step 4: generation. Tokens will be used
        Console.WriteLine(await chatClient.GetResponseAsync(new ChatMessage(ChatRole.User, prompt)));
    }

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

    private static async Task<VectorSearchResults<VectorRecord>> SemanticSearchAsync(
        IVectorStoreRecordCollection<int, VectorRecord> store, 
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
        var searchResults = await store.VectorizedSearchAsync(queryEmbedding, searchOptions);
        return searchResults;
    }

    private static async Task IngestionAsync(
        IVectorStoreRecordCollection<int, VectorRecord> store,
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
            store.UpsertAsync(record);
        }
    }

    private static async Task<IVectorStoreRecordCollection<int, VectorRecord>> GetVectorStoreIfNotExistsThenCreateAsync()
    {
        var vectorStore = new InMemoryVectorStore();
        await vectorStore.GetCollection<int, VectorRecord>(_collectionName, GetRecordDefinition()).CreateCollectionIfNotExistsAsync();
        IVectorStoreRecordCollection<int, VectorRecord> collection = vectorStore.GetCollection<int, VectorRecord>(_collectionName, GetRecordDefinition());

        return vectorStore.GetCollection<int, VectorRecord>(_collectionName, GetRecordDefinition());
    }
    
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

record VectorRecord(int id, string value, ReadOnlyMemory<float> vector, int? productId = null);

