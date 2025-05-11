using Spectre.Console;

while (true)
{
    var prompt =
        AnsiConsole
            .Prompt(
                new SelectionPrompt<string>()
                    .Title("Enter a command")
                    .PageSize(10)
                    .EnableSearch()
                    .MoreChoicesText("[grey](Move up and down to reveal more choices)[/]")
                    .AddChoices(["Choose sample", "Quit"])
            );

    if (prompt == "Quit") break;

    if (prompt == "Choose sample")
    {
        var selectedSample =
            AnsiConsole
                .Prompt(
                    new SelectionPrompt<string>()
                        .Title("Choose a sample")
                        .PageSize(10)
                        .MoreChoicesText("[grey](Move up and down to reveal more choices)[/]")
                        .AddChoices([
                            "Text Embedding",
                            "Text Embedding Ollama",
                            "Text Embedding Kozuchi"
                        ])
                );


        // Execute the selected sample
        await (selectedSample switch
        {
            "Text Embedding" => OpenAiSamples.TextEmbedding(),
            "Text Embedding Ollama" => OpenAiSamples.TextEmbedding_Ollama(),
            "Text Embedding Kozuchi" => OpenAiSamples.TextEmbedding_Kozuchi(),
            _ => Task.CompletedTask,
            
        });
    }
}
