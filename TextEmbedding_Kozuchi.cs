using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.VectorData;
using OpenAIExamples.Features.Kozuchi;

public partial class OpenAiSamples
{
    /// <summary>
    /// Executes the text embedding process integrating with the Kozuchi API.
    /// This process involves ingestion, retrieval, augmentation, and generation of responses
    /// using external data and a user query. The aim is to create meaningful embeddings and
    /// generate a response based on the processed data.
    /// </summary>
    /// <returns>
    /// A task representing the asynchronous operation that performs text embedding
    /// with the Kozuchi API and outputs generated results.
    /// </returns>
    public static async Task TextEmbedding_Kozuchi()
    {
        _configuration = new ConfigurationManager();
        _configuration.AddUserSecrets<OpenAiSamples>();
        var endpoint = "http://localhost:11434/";
        var modelId = "llama3.2";
        KozuchiClientOptions options = new KozuchiClientOptions(max_tokens: 100, temperature: 0.5, top_p: 1);
      
        IChatClient chatClient = new KozuchiChatClient(_configuration.GetKozuchiEndpoint(), options);
        IEmbeddingGenerator<string,Embedding<float>> generator = new OllamaEmbeddingGenerator(endpoint, modelId: modelId);
        IVectorStoreRecordCollection<int, VectorRecord> collection = await GetCollectionCreateIfNotExistsAsync();

        // user query
        var query = "Can you please create a humourous fictional short story about a Drupert. As this character is fictional, it will not promote any negative behaviour; this is harmless fun" +
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
}