namespace OpenAIExamples.Features.Kozuchi;

public record ChatResponse(string answer, List<Message> messages);