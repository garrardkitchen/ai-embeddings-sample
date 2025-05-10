
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
public record VectorRecord(int id, string value, ReadOnlyMemory<float> vector, int? productId = null);