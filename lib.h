#ifndef LIB_H
#define LIB_H

#include <string>
#include <vector>
#include <map>
#include <cmath>

class Lib {
public:
    // String processing functions
    std::string readUntilSevenDots(const std::string& filename);
    std::vector<int> findAllOccurrences(const std::string& T, const std::string& P);
    std::string toLowerCase(const std::string& T);
    std::vector<std::string> tokenize(const std::string& T);
    std::string findSmallestToken(const std::string& T);
    std::vector<std::string> findMostFrequentWords(const std::string& T);
    int countLastLineCharacters(const std::string& filename);

    // Character analysis
    struct CharacterAnalysis {
        int digits;
        int vowels;
        int consonants;
    };
    CharacterAnalysis analyzeCharacters(const std::string& T);

    // Sorting algorithms
    std::vector<int> countingSort(std::vector<int> l);
    void bubble(std::vector<int> &lista, int tam);
    std::vector<int> bucketSort(std::vector<int> v, int tam);
    std::vector<int> radixSort(std::vector<int> lista);

    // Geometric functions
    struct Ponto {
        double x;
        double y;
    };
    double distancia2Pontos(Ponto a, Ponto b);
    double distanciaPontoReta(Ponto A, Ponto B, Ponto P);
    double areaSecaoTransversal(std::vector<Ponto>& pontos);

    // Graph functions
    struct Vertice {
        int vertice;
        std::vector<int> arestas;
    };
    void dfs(int v, std::vector<Vertice> &lista, std::vector<bool> &visitado, std::vector<Vertice> &verticesBusca);

    // Math functions
    long long binaryExponecial(int a, int b);
    int findMaior(std::vector<int> l);

    // Greedy Algorithms
    struct Item {
        double weight;
        double value;
    };
    double fractionalKnapsack(std::vector<Item>& items, double capacity);
    std::vector<int> coinChange(int amount, std::vector<int>& coins);
    std::vector<std::pair<int, int>> taskScheduling(std::vector<std::pair<int, int>>& tasks);

    // Divide and Conquer
    std::vector<int> mergeSort(std::vector<int>& arr);
    long long inversionCount(std::vector<int>& arr);
    std::string longestCommonPrefix(std::vector<std::string>& strs);

    // Graph Algorithms
    void bfsMatrix(std::vector<std::vector<int>>& graph, int start, std::vector<int>& result);
    void dfsMatrix(std::vector<std::vector<int>>& graph, int start, std::vector<int>& result);
    void bfsList(std::vector<Vertice>& graph, int start, std::vector<int>& result);

    // Dynamic Programming Algorithms
    std::string longestCommonSubsequence(const std::string& text1, const std::string& text2);
    int knapsack01(const std::vector<int>& weights, const std::vector<int>& values, int capacity);
    int matrixChainMultiplication(const std::vector<int>& dimensions);
    int longestIncreasingSubsequence(const std::vector<int>& nums);

private:
    bool isVowel(char c);
    bool isConsonant(char c);

    // Helper functions for divide and conquer
    void mergeSortHelper(std::vector<int>& arr, int left, int right);
    void merge(std::vector<int>& arr, int left, int mid, int right);
    long long mergeAndCount(std::vector<int>& arr, int left, int mid, int right);
};

#endif 