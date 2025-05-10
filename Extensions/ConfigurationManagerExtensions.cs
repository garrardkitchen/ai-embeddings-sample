using Microsoft.Extensions.Configuration;

#pragma warning disable
public static class ConfigurationManagerExtensions
{
    /// <summary>
    /// Retrieves the GitHub Models token from the configuration.
    /// </summary>
    /// <param name="configuration">The <see cref="ConfigurationManager"/> instance from which to retrieve the token.</param>
    /// <returns>The GitHub Models token as a string.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the "GitHubModels:Token" configuration value is missing or null.
    /// </exception>
    public static string GetGitHubModelsToken(this ConfigurationManager configuration)
    {
        return configuration.GetValue<string>("GitHubModels:Token") ?? throw new InvalidOperationException("Missing configuration: GitHubModels:Token. Ensure it is set in UserSecrets.");
    }
}