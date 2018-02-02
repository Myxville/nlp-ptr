using System;
using System.Collections.Generic;
using System.Linq;
using System.Security;
using System.Text;

namespace wordRecovery
{
    public class Functions
    {
        public static List<HeapItem> GetOptions(double prefixProba, string prefix, string suffix,
            LanguageModel langModel, MissingLetterModel missModel, double optimism = 0.5,
            Dictionary<int, Double> cache = null)
        {
            var result = new List<HeapItem>();

            var nextLetter = string.Empty;
            var newSuffix = string.Empty;
            var newPrefix = string.Empty;

            var probaMissingState = 0.0d;
            var probaNextLetter = 0.0d;
            var probaSuffix = 0.0d;

            foreach (var vocabularItem in langModel.Vocabulary.Union(new List<string>(){""}))
            {
                if (!string.IsNullOrEmpty(vocabularItem))
                {
                    nextLetter = vocabularItem;
                    newSuffix = suffix;
                    newPrefix = prefix + nextLetter;

                    probaMissingState = -1 * Math.Log(missModel.PredictProba(prefix, nextLetter));
                }
                else
                {
                    nextLetter = suffix[0].ToString();
                    newSuffix = suffix.Substring(1);
                    newPrefix = prefix + nextLetter;

                    probaMissingState = -1 * Math.Log(1 - missModel.PredictProba(prefix, nextLetter));
                }

                probaNextLetter = -1 * Math.Log(langModel.SingleProba(prefix, nextLetter));

                if (cache != null)
                    probaSuffix = cache[newSuffix.Length] * optimism;
                else
                    probaSuffix = -1 * Math.Log(langModel.SingleProba(newPrefix, newSuffix)) * optimism;

                var proba = prefixProba + probaNextLetter + probaMissingState + probaSuffix;
                result.Add(new HeapItem()
                {
                    BestLogProb = proba,
                    Prefix = newPrefix,
                    Suffix = newSuffix,
                    Letter = vocabularItem,
                    BestOptimisticProb = probaSuffix
                });
            }

            return result;
        }

        public static Dictionary<string, double> NoisyChannel(string word, LanguageModel langModel,
            MissingLetterModel missModel, double freedom = 3.0, int maxAttempts = 10000, double optimism = 0.9,
            bool verbose = false)
        {
            var result = new Dictionary<string, double>();

            var heap = new List<HeapItem>();
            var candidates = new List<HeapItem>();

            var query = word + " ";
            var prefix = " ";
            var suffix = query;

            var prefixProba = 0.0d;
            var fullOriginLogProb = -1 * langModel.SingleLogarithmProba(prefix, query);
            var noMissingLogProb = -1 * missModel.SingleLogarithmProba(prefix, query);

            var bestLogProb = fullOriginLogProb + noMissingLogProb;

            heap.Add(new HeapItem()
            {
                BestLogProb = bestLogProb * optimism,
                BestOptimisticProb = bestLogProb * optimism,
                Prefix = prefix,
                Suffix = suffix,
                Letter = string.Empty
            });

            candidates.Add(new HeapItem()
            {
                BestLogProb = bestLogProb,
                Prefix = prefix + query,
                Suffix = string.Empty,
                Letter = string.Empty,
                BestOptimisticProb = 0.0d
            });

            if (verbose)
                Console.WriteLine("Baseline score: {0}", bestLogProb);

            var cache = new Dictionary<int, Double>();
            var futureSuffix = string.Empty;

            for (int i = -1; i < query.Length; i++)
            {
                futureSuffix = query.Substring(0, i + 1);
                cache.Add(futureSuffix.Length, -1 * langModel.SingleLogarithmProba(string.Empty, futureSuffix));
                cache[futureSuffix.Length] += -1 * missModel.SingleLogarithmProba(string.Empty, futureSuffix);
            }

            for (int i = 0; i < maxAttempts; i++)
            {
                if (!heap.Any())
                    break;

                var currentCandidate = heap.Last();
                heap.Remove(currentCandidate);

                if (currentCandidate.Suffix.Equals(string.Empty))
                {
                    if (currentCandidate.BestLogProb <= bestLogProb + freedom)
                    {
                        candidates.Add(currentCandidate);
                        if (currentCandidate.BestLogProb < bestLogProb)
                            bestLogProb = currentCandidate.BestLogProb;
                    }
                }
                else
                {
                    prefixProba = currentCandidate.BestLogProb - currentCandidate.BestOptimisticProb;
                    prefix = currentCandidate.Prefix;
                    suffix = currentCandidate.Suffix;

                    var options = GetOptions(prefixProba, prefix, suffix, langModel, missModel, optimism, cache);
                    heap.AddRange(options.Where(option => option.BestLogProb < bestLogProb + freedom));
                }

            }

            foreach (var candidate in candidates)
            {
                if (candidate.BestLogProb <= bestLogProb + freedom)
                {
                    if (result.ContainsKey(candidate.Prefix))
                        result[candidate.Prefix] = Math.Min(result[candidate.Prefix], candidate.BestLogProb);
                    else
                        result.Add(candidate.Prefix, candidate.BestLogProb);
                }
            }

            return new Dictionary<string, double>(result.OrderBy(x => x.Value).Take(3));
        }
    }

    public class HeapItem
    {
        public double BestLogProb { get; set; }
        public string Prefix { get; set; }
        public string Suffix { get; set; }
        public string Letter { get; set; }
        public double BestOptimisticProb { get; set; }
    }
}
