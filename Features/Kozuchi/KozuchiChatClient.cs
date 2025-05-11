using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;

namespace OpenAIExamples.Features.Kozuchi;

public class KozuchiChatClient : IChatClient
{
    private HttpClient _client;
    private readonly ConfigurationManager _configuration;
    private readonly Uri _url;
    private readonly KozuchiClientOptions _options;

    /// <summary>
    /// Provides an implementation of the IChatClient interface for interacting with the Kozuchi conversational AI platform.
    /// Responsible for configuring and managing requests, including handling API endpoint configuration
    /// and managing default settings for KozuchiChatClient operations.
    /// </summary>
    public KozuchiChatClient(string endpoint, KozuchiClientOptions? options)
    {
        _client = new HttpClient();
        _configuration = new ConfigurationManager();
        _configuration.AddUserSecrets<OpenAiSamples>();
        _client.DefaultRequestHeaders.CacheControl = CacheControlHeaderValue.Parse("no-cache");
        _client.DefaultRequestHeaders.Add("Ocp-Apim-Subscription-Key", _configuration.GetKozuchiToken());
        _url =  new Uri($"{endpoint}conversational-generative-ai/api/v1/action/defined/text:simple_chat/call");
        if (options == null)
        {
            _options = new KozuchiClientOptions(null, max_tokens: 1, temperature:0.5, top_p: 1);
        }
        else
        {
            _options = options;
        }
    }
    
    public void Dispose()
    {
        _configuration.Dispose();
        _client.Dispose();
    }

    /// <summary>
    /// Sends a request to the Kozuchi conversational AI platform to generate a response
    /// based on the provided chat messages and options. Returns the generated response as a ChatResponse object.
    /// </summary>
    /// <param name="chatMessages">The collection of chat messages to be sent as input to the Kozuchi AI.</param>
    /// <param name="options">Optional configuration for the chat request, such as model settings and token limits.</param>
    /// <param name="cancellationToken">A cancellation token to observe while waiting for the task to complete.</param>
    /// <returns>A task representing the asynchronous operation that returns a ChatResponse containing the AI-generated result.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the response from Kozuchi cannot be deserialized or is invalid.</exception>
    public async Task<Microsoft.Extensions.AI.ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> chatMessages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = new CancellationToken())
    {
        var messages = chatMessages.ToKozuchiMessages();
        
        HttpResponseMessage response;
        ChatRequest request = new ChatRequest(messages?.Last().content, messages);
        request = request with {
            max_tokens = _options.max_tokens, 
            temperature = _options.temperature,
            top_p = _options.top_p,
            model =  _options.model
        };

        using (var content = new StringContent(JsonSerializer.Serialize(request), Encoding.UTF8, "application/json"))
        {
            content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
            response = await _client.PostAsync(_url, content);
        }

        var responseStream = await response.Content.ReadAsStreamAsync();
        var result = await JsonSerializer.DeserializeAsync<OpenAIExamples.Features.Kozuchi.ChatResponse>(responseStream) ?? throw new InvalidOperationException("Failed to deserialize response.");
        return result.ToChatResponse();
    }

    public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        CancellationToken cancellationToken = new CancellationToken())
    {
        throw new NotImplementedException();
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        throw new NotImplementedException();
    }
}