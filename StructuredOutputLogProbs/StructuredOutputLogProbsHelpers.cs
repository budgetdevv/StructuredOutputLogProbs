using System.Buffers;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
#if DEBUG
using System.Diagnostics;
#endif

namespace StructuredOutputLogProbs
{
    public static class StructuredOutputLogProbsHelpers
    {
        public readonly struct TokenLogProb(double logProbability, ReadOnlyMemory<byte> tokenUtf8Bytes)
        {
            public readonly double LogProbability = logProbability;

            public readonly ReadOnlyMemory<byte> TokenUTF8Bytes = tokenUtf8Bytes;
        }

        public readonly struct FieldProbability(double jointProbability, double averageProbability)
        {
            public readonly double JointProbability = jointProbability;

            public readonly double AverageProbability = averageProbability;
        }

        public static void GetFieldProbabilities(
            ReadOnlySpan<char> jsonText,
            IList<TokenLogProb> logProbs,
            out Dictionary<string, FieldProbability> fieldProbabilities)
        {
            fieldProbabilities = new();

            GetFieldProbabilities(jsonText, logProbs, fieldProbabilities);
        }

        public static void GetFieldProbabilities(
            ReadOnlySpan<char> jsonText,
            IList<TokenLogProb> logProbs,
            Dictionary<string, FieldProbability> fieldProbabilities)
        {
            var utf8Encoding = Encoding.UTF8;

            var arrayPool = ArrayPool<byte>.Shared;

            var jsonUTF8ByteCount = utf8Encoding.GetByteCount(jsonText);

            var jsonUTF8BytesBuffer = arrayPool.Rent(jsonUTF8ByteCount);

            var jsonUTF8BytesSpan = MemoryMarshal.CreateSpan(
                ref MemoryMarshal.GetArrayDataReference(jsonUTF8BytesBuffer),
                jsonUTF8ByteCount
            );

            var getBytesMatch = jsonUTF8ByteCount == utf8Encoding.GetBytes(
                jsonText,
                jsonUTF8BytesSpan
            );

            #if DEBUG
            Debug.Assert(
                getBytesMatch,
                "UTF-8 encoding of the JSON text does not match the expected byte count."
            );
            #endif

            var jsonReader = new Utf8JsonReader(jsonUTF8BytesSpan);

            var currentLogProbsIndex = 0;

            var logProbsConsumedBytes = 0;

            var previousTokenLogProbValue = 0.0;

            while (jsonReader.Read())
            {
                var tokenType = jsonReader.TokenType;

                if (tokenType != JsonTokenType.PropertyName)
                {
                    continue;
                }

                var currentFieldName = jsonReader.GetString()!;

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

                while (logProbsConsumedBytes < valueStartIndex)
                {
                    var currentTokenLogProb = logProbs[currentLogProbsIndex++];

                    previousTokenLogProbValue = currentTokenLogProb.LogProbability;

                    var decodedLength = currentTokenLogProb.TokenUTF8Bytes.Length;

                    logProbsConsumedBytes += decodedLength;
                }

                // Are we exactly aligned?
                var isAligned = logProbsConsumedBytes == valueStartIndex;

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

                    var decodedLength = currentTokenLogProb.TokenUTF8Bytes.Length;

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

            // Even in the event where an exception is thrown,
            // we do not leak memory, as the GC collects the rented array.
            // We avoid wrapping this in try-finally, as there is some performance overhead.
            arrayPool.Return(jsonUTF8BytesBuffer);
        }
    }
}