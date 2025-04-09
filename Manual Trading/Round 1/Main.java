import java.util.*;

public class Main {

    // Define the ordering for our conversion table.
    // These arrays reflect the table’s row and column order.
    static String[] fromCurrencies = { "SeaShells", "Pizza's", "Snowballs", "Silicon Nuggets" };
    static String[] toCurrencies   = { "Snowballs", "Pizza's", "Silicon Nuggets", "SeaShells" };

    // Conversion rates table.
    // Row i corresponds to fromCurrencies[i] and column j corresponds to toCurrencies[j].
    static double[][] exchangeRate = {
        //                  To:   Snowballs   Pizza's   Silicon Nuggets   SeaShells
        /*From SeaShells*/      {1.34,       1.98,       0.64,            1},
        /*From Pizza's*/        {0.7,        1,          0.31,            0.48},
        /*From Snowballs*/      {1,          1.45,       0.52,            0.72},
        /*From Silicon Nuggets*/{1.95,       3.1,        1,               1.49}
    };

    // Helper: returns conversion rate from currency "from" to "to" using the proper ordering.
    public static double getRate(String from, String to) {
        int fromIndex = -1;
        int toIndex = -1;
        for (int i = 0; i < fromCurrencies.length; i++) {
            if (fromCurrencies[i].equals(from)) {
                fromIndex = i;
                break;
            }
        }
        for (int j = 0; j < toCurrencies.length; j++) {
            if (toCurrencies[j].equals(to)) {
                toIndex = j;
                break;
            }
        }
        if (fromIndex == -1 || toIndex == -1) {
            throw new IllegalArgumentException("Currency not found: " + from + " or " + to);
        }
        return exchangeRate[fromIndex][toIndex];
    }
    
    public static void main(String[] args) {
        double initialAmount = 500.0;
        String startCurrency = "SeaShells";  // We always start (and must end) with SeaShells.
        
        // Map to record each unique strategy and its final SeaShell total.
        Map<String, Double> strategyMap = new LinkedHashMap<>();
        String bestStrategy = "";
        double bestValue = 0.0;
        
        // --- 2 trades ---
        // Trade 1: SeaShells -> A, with A ≠ SeaShells
        // Trade 2: A -> SeaShells (A must differ from SeaShells by definition)
        for (String A : fromCurrencies) {
            if (A.equals(startCurrency)) continue;
            double amt1 = initialAmount * getRate(startCurrency, A);
            double finalAmt = amt1 * getRate(A, startCurrency);
            String strategy = startCurrency + " -> " + A + " | " + A + " -> " + startCurrency + "  [2 trades]";
            strategyMap.put(strategy, finalAmt);
            if (finalAmt > bestValue) {
                bestValue = finalAmt;
                bestStrategy = strategy;
            }
        }
        
        // --- 3 trades ---
        // Trade 1: SeaShells -> A (A ≠ SeaShells)
        // Trade 2: A -> B, with B ≠ A and B ≠ SeaShells (to allow a valid conversion in Trade 3)
        // Trade 3: B -> SeaShells (B ≠ SeaShells)
        for (String A : fromCurrencies) {
            if (A.equals(startCurrency)) continue;
            for (String B : fromCurrencies) {
                if (B.equals(A) || B.equals(startCurrency)) continue;
                double amt1 = initialAmount * getRate(startCurrency, A);
                double amt2 = amt1 * getRate(A, B);
                double finalAmt = amt2 * getRate(B, startCurrency);
                String strategy = startCurrency + " -> " + A + " | " + A + " -> " + B + " | " + B + " -> " + startCurrency + "  [3 trades]";
                strategyMap.put(strategy, finalAmt);
                if (finalAmt > bestValue) {
                    bestValue = finalAmt;
                    bestStrategy = strategy;
                }
            }
        }
        
        // --- 4 trades ---
        // Trade 1: SeaShells -> A (A ≠ SeaShells)
        // Trade 2: A -> B, with B ≠ A (B can be any, even SeaShells, but then trade 3 must change currency)
        // Trade 3: B -> C, with C ≠ B and also C ≠ SeaShells (since Trade 4 is forced: C -> SeaShells)
        // Trade 4: C -> SeaShells
        for (String A : fromCurrencies) {
            if (A.equals(startCurrency)) continue;
            for (String B : fromCurrencies) {
                if (B.equals(A)) continue;
                for (String C : fromCurrencies) {
                    if (C.equals(B) || C.equals(startCurrency)) continue;
                    double amt1 = initialAmount * getRate(startCurrency, A);
                    double amt2 = amt1 * getRate(A, B);
                    double amt3 = amt2 * getRate(B, C);
                    double finalAmt = amt3 * getRate(C, startCurrency);
                    String strategy = startCurrency + " -> " + A + " | " + A + " -> " + B + " | " + B + " -> " + C + " | " + C + " -> " + startCurrency + "  [4 trades]";
                    strategyMap.put(strategy, finalAmt);
                    if (finalAmt > bestValue) {
                        bestValue = finalAmt;
                        bestStrategy = strategy;
                    }
                }
            }
        }
        
        // --- 5 trades ---
        // Trade 1: SeaShells -> A (A ≠ SeaShells)
        // Trade 2: A -> B, with B ≠ A
        // Trade 3: B -> C, with C ≠ B
        // Trade 4: C -> D, with D ≠ C and D ≠ SeaShells (so that Trade 5 is valid)
        // Trade 5: D -> SeaShells
        for (String A : fromCurrencies) {
            if (A.equals(startCurrency)) continue;
            for (String B : fromCurrencies) {
                if (B.equals(A)) continue;
                for (String C : fromCurrencies) {
                    if (C.equals(B)) continue;
                    for (String D : fromCurrencies) {
                        if (D.equals(C) || D.equals(startCurrency)) continue;
                        double amt1 = initialAmount * getRate(startCurrency, A);
                        double amt2 = amt1 * getRate(A, B);
                        double amt3 = amt2 * getRate(B, C);
                        double amt4 = amt3 * getRate(C, D);
                        double finalAmt = amt4 * getRate(D, startCurrency);
                        String strategy = startCurrency + " -> " + A + " | " + A + " -> " + B + " | " + B + " -> " + C + " | " + C + " -> " + D + " | " + D + " -> " + startCurrency + "  [5 trades]";
                        strategyMap.put(strategy, finalAmt);
                        if (finalAmt > bestValue) {
                            bestValue = finalAmt;
                            bestStrategy = strategy;
                        }
                    }
                }
            }
        }
        
        // Print the total number of strategies computed.
        System.out.println("Total strategies evaluated: " + strategyMap.size());
        System.out.println("\nAll strategies and their final SeaShell totals:");
        for (Map.Entry<String, Double> entry : strategyMap.entrySet()) {
            System.out.printf("%s : %.4f SeaShells%n", entry.getKey(), entry.getValue());
        }
        
        // Print the best overall strategy.
        System.out.println("\nBest Strategy:");
        System.out.println(bestStrategy);
        System.out.printf("Maximum SeaShells: %.4f%n", bestValue);
    }
}
