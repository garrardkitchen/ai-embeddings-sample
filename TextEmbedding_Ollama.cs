using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.VectorData;

public partial class OpenAiSamples
{
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
    public static async Task TextEmbedding_Ollama()
    {
        _configuration = new ConfigurationManager();
        _configuration.AddUserSecrets<OpenAiSamples>();
        var endpoint = "http://localhost:11434/";
        var modelId = "llama3.2:1b";
      
        IChatClient chatClient = new OllamaChatClient(endpoint, modelId: modelId);
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

