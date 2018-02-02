using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;

namespace wordRecovery
{
    public class MissingLetterModel
    {
        public int Order { get; set; }
        public double SmoothingMissing { get; set; }
        public double SmoothingTotal { get; set; }

        public Dictionary<string, Dictionary<string, double>> Missed { get; set; }
        public Dictionary<string, Dictionary<string, double>> Total { get; set; }

        public MissingLetterModel()
        {
            Order = 0;
            SmoothingMissing = 0.1d;
            SmoothingTotal = 1.0d;

            Missed = new Dictionary<string, Dictionary<string, double>>();
            Total = new Dictionary<string, Dictionary<string, double>>();
        }

        public void Initialize(List<KeyValuePair<string, string>> data)
        {
            foreach (var pair in data)
            {
                for (int i = 0; i < pair.Key.Length; i++)
                {
                    var origLetter = pair.Key.Substring(i, 1 + Order);
                    var observedLetter = pair.Value.Substring(i, 1 + Order);

                    var context = pair.Key.Substring(i, Order);

                    if (observedLetter.Equals("-"))
                    {
                        if (Missed.ContainsKey(context))
                        {
                            if (Missed[context].ContainsKey(origLetter))
                                Missed[context][origLetter] += 1;
                            else
                                Missed[context].Add(origLetter, 1);
                        }
                        else
                        {
                            Missed.Add(context, new Dictionary<string, double>());
                            Missed[context].Add(origLetter, 1);
                        }
                    }

                    if (Total.ContainsKey(context))
                    {
                        if (Total[context].ContainsKey(origLetter))
                            Total[context][origLetter] += 1;
                        else
                            Total[context].Add(origLetter, 1);
                    }
                    else
                    {
                        Total.Add(context, new Dictionary<string, double>());
                        Total[context].Add(origLetter, 1);
                    }
                }
            }
        }

        public double PredictProba(string context, string lastLetter)
        {
            var local = string.Empty;
            if (Order > 0)
                local = context.Substring(context.Length - 2);

            var missedFreq = SmoothingMissing;
            if (Missed.ContainsKey(local))
                if (Missed[local].ContainsKey(lastLetter))
                    missedFreq += Missed[local][lastLetter];

            var totalFreq = SmoothingTotal;
            if (Total.ContainsKey(local))
                if (Total[local].ContainsKey(lastLetter))
                    totalFreq += Total[local][lastLetter];

            return missedFreq / totalFreq;
        }

        public double SingleLogarithmProba(string context, string continuation, string actual = "")
        {
            if (string.IsNullOrEmpty(actual))
                actual = continuation;

            var result = 0.0d;

            for (int i = 0; i < continuation.Length; i++)
            {
                var origToken = continuation.Substring(i, 1);
                var actToken = actual.Substring(i, 1);

                var pp = PredictProba(context, origToken);
                if (!actToken.Equals("-"))
                    pp = 1 - pp;

                result += Math.Log(pp);

                context += origToken;
            }

            return result;
        }

        public double SingleProba(string context, string continuation, string actual = "")
        {
            return Math.Exp(SingleProba(context, continuation, actual));
        }
    }
}
