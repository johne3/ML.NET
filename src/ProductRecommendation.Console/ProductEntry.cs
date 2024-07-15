using Microsoft.ML.Data;

namespace ProductRecommendation.Console;

internal class ProductEntry
{
    [KeyType(count: 262111)]
    public uint ProductID { get; set; }

    [KeyType(count: 262111)]
    public uint CoPurchaseProductID { get; set; }
}
