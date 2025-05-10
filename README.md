# Microsoft.Extensions.AI - OpenAI Example

This project contains a set of samples that show how to use the OpenAI reference implementation in the [Microsoft.Extensions.AI.OpenAI NuGet package](https://aka.ms/meai-openai-nuget).

## Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
- [Visual Studio](https://visualstudio.microsoft.com/downloads/) or [VS Code](https://visualstudio.microsoft.com/downloads/)
- An Open AI API key. For more details, see the [OpenAI documentation](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key).

## Setup

1. Create user secret:

Your PAT requires no scope to run this example and this is only required for the Non Ollama models.

```bash
dotnet user-secrets set "GitHubModels:Token" "<your-token-value>"
```

## Quick Start

### Visual Studio

1. Open the *ai-embeddings-sample.csproj* project
1. Press <kbd>F5</kbd>.

### Visual Studio Code

1. Open your terminal
1. Navigate to the *ai-embeddings-sample* project directory
1. Run the applicaton using `dotnet run`

    ```dotnetcli
    dotnet run
    ```
## Test your application

1. When the application starts, select **Choose sample**.
1. Select one of the samples from the dropdown to run it. 
1. After the selected sample runs, you can choose to run another sample or select **Quit** to stop the application.

## Examples

| Example                                            | Description                                         |
|----------------------------------------------------|-----------------------------------------------------|
| [Text Embedding](./TextEmbedding.cs)               | Use text embedding generator                        |
| [Text Embedding Ollama](./TextEmbedding_Ollama.cs) | Use text embedding generator using Ollama generator |

## References

- Based off of [dotnet/ai-samples](https://github.com/dotnet/ai-samples)

## Known Issues

- You may only be able to run the Ollama embedding example once before it responds with this:

>[!Warning]
>I can't create content that promotes or encourages harmful behavior, such 
as telling off someone for laughing during class time. Can I help you with something else?

   **What to do**:
   - Just restart the sample with `dotnet run` and it should work again.
