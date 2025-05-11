using System.IO.Compression;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;

namespace OpenAIExamples.Features.Kozuchi;

public static class KozuchiExtensions
{
    /// <summary>
    /// Converts an <see cref="IEnumerable{ChatMessage}"/> instance
    /// to a list of <see cref="Message"/> objects compatible with Kozuchi.
    /// </summary>
    /// <param name="chatMessages">The collection of <see cref="ChatMessage"/> objects to be converted.</param>
    /// <returns>A list of <see cref="Message"/> objects containing the converted data.</returns>
    public static List<Message> ToKozuchiMessages(this IEnumerable<ChatMessage> chatMessages)
    {
        List<Message> list = new List<Message>();
        foreach (var message in chatMessages)
        {
            list.Add(new Message(message.Role.ToString(), message.Text));
        }
        return list;
    }

    /// <summary>
    /// Converts a <see cref="OpenAIExamples.Features.Kozuchi.ChatResponse"/> instance
    /// to a <see cref="Microsoft.Extensions.AI.ChatResponse"/> instance.
    /// </summary>
    /// <param name="chatResponses">The <see cref="OpenAIExamples.Features.Kozuchi.ChatResponse"/> object to be converted.</param>
    /// <returns>A <see cref="Microsoft.Extensions.AI.ChatResponse"/> object containing the converted data.</returns>
    public static Microsoft.Extensions.AI.ChatResponse ToChatResponse(this OpenAIExamples.Features.Kozuchi.ChatResponse chatResponses)
    {
        return new Microsoft.Extensions.AI.ChatResponse(new ChatMessage(ChatRole.Assistant, chatResponses.answer));
    }
}