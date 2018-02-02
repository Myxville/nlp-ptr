using System;
using System.Collections.Generic;
using System.Linq;

namespace wordRecovery
{
    public class LanguageModel
    {
        public int Order { get; set; }
        public double Smoothing { get; set; }
        public double Recursive { get; set; }
        public LanguageModel Child { get; set; }
        public List<string> Vocabulary { get; set; }
        public Dictionary<string, Dictionary<string, double>> Data { get; set; }
        public Dictionary<string, double> Series { get; set; }

        public LanguageModel()
        {
            Order = 4;
            Smoothing = 0.001d;
            Recursive = 0.01d;
            Vocabulary = new List<string>();
            Data = new Dictionary<string, Dictionary<string, double>>();
            Series = new Dictionary<string, double>();
        }

        public void Initialize(string data)
        {
            for (int i = 0; i < data.Length - Order; i++)
            {
                var token = data.Substring(i + Order, 1);
                var context = data.Substring(i, Order);

                if (Data.ContainsKey(context))
                {
                    var itemDic = Data[context];
                    if (itemDic.ContainsKey(token))
                    {
                        itemDic[token]++;
                    }
                    else
                    {
                        itemDic.Add(token, 1);
                    }
                }
                else
                {
                    Data.Add(context, new Dictionary<string, double>());
                    Data[context].Add(token, 1);
                }

                if (!Vocabulary.Contains(token))
                    Vocabulary.Add(token);
            }

            Vocabulary.Sort();

            if (Recursive > 0 && Order > 0)
            {
                Child = new LanguageModel {Order = Order - 1, Recursive = Recursive, Smoothing = Smoothing};
                Child.Initialize(data);
            }
        }

        public Dictionary<string, double> GetCounts(string context)
        {
            var local = string.Empty;
            if (Order > 0)
            {
                var startIndex = context.Length - Order;
                if (startIndex < 0)
                    startIndex = 0;

                local = context.Substring(startIndex, Math.Min(Order, context.Length));
            }

            var freqDict = new Dictionary<string, double>();
            
            if (Data.ContainsKey(local))
                freqDict = Data[local];

            var freq = Vocabulary.ToDictionary(x => x, x => 0.0d);
            foreach (string t in Vocabulary)
            {
                var freqDictValue = 0.0d;
                if (freqDict.ContainsKey(t))
                    freqDictValue = freqDict[t];

                if (freq.ContainsKey(t))
                    freq[t] = freqDictValue + Smoothing;
                else
                {
                    freq.Add(t, freqDictValue + Smoothing);
                }
            }

            if (Recursive > 0 && Order > 0)
            {
                var childFreq = Child.GetCounts(context);
                foreach (var f in childFreq.Keys)
                {
                    if (freq.ContainsKey(f))
                        freq[f] += childFreq[f] * Recursive;
                    else
                        freq.Add(f, childFreq[f] * Recursive);
                }
            }

            return freq;
        }

        public Dictionary<string, double> PredictProba(string context)
        {
            var counts = GetCounts(context);
            var sum = counts.Keys.Select(x => counts[x]).Sum();

            var preCounts = counts.ToDictionary(x => x.Key, x => x.Value);

            foreach (var key in preCounts.Keys)
            {
                counts[key] = counts[key] / sum;
            }

            return counts;
        }

        public double SingleLogarithmProba(string context, string continuation)
        {
            var result = 0.0d;
            for (int i = 0; i < continuation.Length; i++)
            {
                var token = continuation.Substring(i, 1);
                result += Math.Log(PredictProba(context)[token]);
                context += token;
            }

            return result;
        }

        public double SingleProba(string context, string continuation)
        {
            return Math.Exp(SingleLogarithmProba(context, continuation));
        }
    }
}
