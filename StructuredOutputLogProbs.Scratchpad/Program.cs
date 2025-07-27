using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace StructuredOutputLogProbs.Scratchpad
{
    internal static class Program
    {
        private struct LogProbs
        {
            public double LogProbability { get; set; }

            [JsonPropertyName("Utf8Bytes")]
            public string Base64EncodedUTF8Bytes { get; set; }

            public string Token { get; set; }

            public int GetDecodedLength()
            {
                var base64EncodedUTF8Bytes = Base64EncodedUTF8Bytes;

                var base64LengthMinusOne = base64EncodedUTF8Bytes.Length - 1;

                int padCount;

                {
                    int i;

                    for (i = base64LengthMinusOne; i >= 0; i--)
                    {
                        if (base64EncodedUTF8Bytes[i] != '=')
                        {
                            break;
                        }
                    }

                    padCount = base64LengthMinusOne - i;
                }

                var decodedLength = (base64EncodedUTF8Bytes.Length * 3 / 4) - padCount;

                #if DEBUG
                Debug.Assert(
                    Convert.FromBase64String(base64EncodedUTF8Bytes).Length == decodedLength,
                    "Base64 decode length does not match expected length."
                );
                #endif

                return decodedLength;
            }
        }

        private struct LLMResponse
        {
            public string Content { get; set; }

            public LogProbs[] ContentTokenLogProbabilities { get; set; }
        }

        private readonly struct FieldProbability(double jointProbability, double averageProbability)
        {
            public readonly double JointProbability = jointProbability;

            public readonly double AverageProbability = averageProbability;
        }

        private const string JSON_FILE_PATH = "llmresponse.json";

        private static void Main(string[] args)
        {
            TestLib();
        }

        private static void TestLib()
        {
            var response = JsonSerializer.Deserialize<LLMResponse>(
                File.ReadAllText(JSON_FILE_PATH)
            );

            var jsonText = response.Content;

            var logProbs = response
                .ContentTokenLogProbabilities
                .Select(x => new StructuredOutputLogProbs.TokenLogProb(
                    x.LogProbability,
                    Convert.FromBase64String(x.Base64EncodedUTF8Bytes)
                ))
                .ToArray();

            StructuredOutputLogProbs.GetFieldProbabilities(
                jsonText,
                logProbs,
                out var fieldProbs
            );

            Console.WriteLine("Field Probabilities:");

            foreach (var (fieldName, probability) in fieldProbs)
            {
                Console.WriteLine(
                $"""
                Current Field: {fieldName}

                Joint Probability: {probability.JointProbability:F6}

                Average Probability: {probability.AverageProbability:F6}
                """);
            }
        }

        private static void Experimentation()
        {
            var response = JsonSerializer.Deserialize<LLMResponse>(
                File.ReadAllText(JSON_FILE_PATH)
            );

            var jsonText = response.Content;

            var logProbs = response.ContentTokenLogProbabilities;

            Console.WriteLine(jsonText);

            Console.WriteLine(logProbs.Length);

            var textReconstructed = new StringBuilder();

            foreach (var logProb in logProbs)
            {
                textReconstructed.Append(logProb.Token);
            }

            Console.WriteLine(textReconstructed);

            var currentLogProbsIndex = 0;

            var logProbsConsumedBytes = 0;

            var previousTokenLogProbValue = 0.0;

            var fieldProbabilities = new Dictionary<string, FieldProbability>();

            var jsonUTF8BytesSpan = Encoding.UTF8.GetBytes(jsonText).AsSpan();

            var jsonReader = new Utf8JsonReader(jsonUTF8BytesSpan);

            while (jsonReader.Read())
            {
                var tokenType = jsonReader.TokenType;

                if (tokenType != JsonTokenType.PropertyName)
                {
                    continue;
                }

                var currentFieldName = jsonReader.GetString()!;

                Console.WriteLine($"Property: {currentFieldName}");

                jsonReader.Skip();

                tokenType = jsonReader.TokenType;

                var valueStartIndex = jsonReader.TokenStartIndex;

                switch (tokenType)
                {
                    case JsonTokenType.String:
                    case JsonTokenType.Number:
                    case JsonTokenType.True:
                    case JsonTokenType.False:
                    case JsonTokenType.Null:
                        goto HandleSingleValue;
                    case JsonTokenType.StartObject:
                    case JsonTokenType.StartArray:
                        goto HandleContinuousValue;
                    default:
                        // Unsupported
                        continue;
                }

                HandleContinuousValue:
                jsonReader.Skip();

                HandleSingleValue:
                var valueLength = jsonReader.BytesConsumed - valueStartIndex;

                var valueSpan = jsonUTF8BytesSpan.Slice((int) valueStartIndex, (int) valueLength);

                // var valueText = Encoding.UTF8.GetString(valueSpan);
                //
                // if (tokenType== JsonTokenType.String)
                // {
                //     Console.WriteLine($"{valueText} ( String )");
                // }
                //
                // else if (tokenType == JsonTokenType.Number)
                // {
                //     Console.WriteLine($"{valueText} ( Number )");
                // }
                //
                // else if (tokenType is JsonTokenType.True or JsonTokenType.False)
                // {
                //     Console.WriteLine($"{valueText} ( Boolean )");
                // }
                //
                // else if (tokenType == JsonTokenType.Null)
                // {
                //     Console.WriteLine("Null value");
                // }

                while (logProbsConsumedBytes < valueStartIndex)
                {
                    var currentTokenLogProb = logProbs[currentLogProbsIndex++];

                    previousTokenLogProbValue = currentTokenLogProb.LogProbability;

                    var decodedLength = currentTokenLogProb.GetDecodedLength();

                    logProbsConsumedBytes += decodedLength;
                }

                // Are we exactly aligned?
                var isAligned = logProbsConsumedBytes == valueStartIndex;

                Console.WriteLine($"Is Aligned: {isAligned}");

                var currentValueCumulativeLogProb = isAligned ?
                    0.0 :
                    previousTokenLogProbValue;

                // If it is aligned, we start from a count of 0.
                // Otherwise, if the previous logprob is a non-zero value,
                // we consider it as part of the count.
                var cumulativeLogProbCount = isAligned || previousTokenLogProbValue == 0 ?
                    0 :
                    1;

                var nextValueBoundaryIndex = valueStartIndex + valueLength;

                while (logProbsConsumedBytes < nextValueBoundaryIndex)
                {
                    var currentTokenLogProb = logProbs[currentLogProbsIndex++];

                    var decodedLength = currentTokenLogProb.GetDecodedLength();

                    logProbsConsumedBytes += decodedLength;

                    var currentTokenLogProbValue = currentTokenLogProb.LogProbability;

                    if (currentTokenLogProbValue != 0)
                    {
                        currentValueCumulativeLogProb += currentTokenLogProbValue;

                        cumulativeLogProbCount++;
                    }
                }

                var currentValueJointProb = Math.Exp(currentValueCumulativeLogProb);

                var currentValueAverageProb = cumulativeLogProbCount != 0
                    ? Math.Exp(currentValueCumulativeLogProb / cumulativeLogProbCount)
                    : 0.0;

                fieldProbabilities.Add(
                    currentFieldName,
                    new(currentValueJointProb, currentValueAverageProb)
                );
            }

            Console.WriteLine("Field Probabilities:");

            foreach (var (fieldName, probability) in fieldProbabilities)
            {
                Console.WriteLine(
                $"""
                Current Field: {fieldName}

                Joint Probability: {probability.JointProbability:F6}

                Average Probability: {probability.AverageProbability:F6}
                """);
            }
        }
    }
}