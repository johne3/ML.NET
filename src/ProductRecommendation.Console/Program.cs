using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using ProductRecommendation.Console;

//Build Model
var trainingDataPath = Path.Combine(Environment.CurrentDirectory, "data", "Amazon0302.txt");

var mLContext = new MLContext();

var trainData = mLContext.Data.LoadFromTextFile(path: trainingDataPath,
    columns: [
        new TextLoader.Column("Label", DataKind.Single, 0),
        new TextLoader.Column(name: nameof(ProductEntry.ProductID), dataKind: DataKind.UInt32, source: [new TextLoader.Range(0)], keyCount: new KeyCount(262111)),
        new TextLoader.Column(name: nameof(ProductEntry.CoPurchaseProductID), dataKind: DataKind.UInt32, source:[new TextLoader.Range(1)], keyCount: new KeyCount(262111))
    ],
    hasHeader: true,
    separatorChar: '\t');

var options = new MatrixFactorizationTrainer.Options
{
    MatrixColumnIndexColumnName = nameof(ProductEntry.ProductID),
    MatrixRowIndexColumnName = nameof(ProductEntry.CoPurchaseProductID),
    LabelColumnName = "Label",
    LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass,
    Alpha = 0.01,
    Lambda = 0.025
};

var est = mLContext.Recommendation().Trainers.MatrixFactorization(options);

//Train Model
var model = est.Fit(trainData);

//Consume Model
var predictionEngine = mLContext.Model.CreatePredictionEngine<ProductEntry, CoPurchasePrediction>(model);

//Once the prediction engine has been created you can predict scores of two products being co-purchased.
var prediction = predictionEngine.Predict(new ProductEntry
{
    ProductID = 3,
    CoPurchaseProductID = 63
});

Console.WriteLine($"Prediction Score: {prediction.Score}");
