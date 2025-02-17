#include "lib.h"
#include <fstream>
#include <algorithm>
#include <sstream>
#include <cctype>
#include <iostream>
#include <cmath>

// String processing functions
std::string Lib::readUntilSevenDots(const std::string& filename) {
    std::ifstream file(filename);
    std::string result, line;
    
    while (std::getline(file, line)) {
        if (line.substr(0, 7) == ".......") break;
        if (!result.empty()) result += " ";
        result += line;
    }
    
    return result;
}

std::vector<int> Lib::findAllOccurrences(const std::string& T, const std::string& P) {
    std::vector<int> positions;
    size_t pos = T.find(P);
    
    while (pos != std::string::npos) {
        positions.push_back(pos);
        pos = T.find(P, pos + 1);
    }
    
    return positions.empty() ? std::vector<int>{-1} : positions;
}

Lib::CharacterAnalysis Lib::analyzeCharacters(const std::string& T) {
    CharacterAnalysis analysis = {0, 0, 0};
    
    for (char c : T) {
        if (std::isdigit(c)) analysis.digits++;
        else if (isVowel(c)) analysis.vowels++;
        else if (isConsonant(c)) analysis.consonants++;
    }
    
    return analysis;
}

std::string Lib::toLowerCase(const std::string& T) {
    std::string result = T;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::vector<std::string> Lib::tokenize(const std::string& T) {
    std::vector<std::string> tokens;
    std::stringstream ss(toLowerCase(T));
    std::string token;
    
    while (ss >> token) {
        token.erase(std::remove(token.begin(), token.end(), '.'), token.end());
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    
    std::sort(tokens.begin(), tokens.end());
    return tokens;
}

std::string Lib::findSmallestToken(const std::string& T) {
    auto tokens = tokenize(T);
    return tokens.empty() ? "" : tokens[0];
}

std::vector<std::string> Lib::findMostFrequentWords(const std::string& T) {
    auto tokens = tokenize(T);
    std::map<std::string, int> frequency;
    int maxFreq = 0;
    
    for (const auto& token : tokens) {
        maxFreq = std::max(maxFreq, ++frequency[token]);
    }
    
    std::vector<std::string> mostFrequent;
    for (const auto& pair : frequency) {
        if (pair.second == maxFreq) {
            mostFrequent.push_back(pair.first);
        }
    }
    
    return mostFrequent;
}

int Lib::countLastLineCharacters(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    std::string lastLine;
    
    bool foundSeven = false;
    while (std::getline(file, line)) {
        if (foundSeven) {
            lastLine = line;
        }
        if (line.substr(0, 7) == ".......") {
            foundSeven = true;
        }
    }
    
    return lastLine.length();
}

bool Lib::isVowel(char c) {
    c = std::tolower(c);
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
}

bool Lib::isConsonant(char c) {
    return std::isalpha(c) && !isVowel(c);
}

// Sorting algorithms
int Lib::findMaior(std::vector<int> l) {
    int maior = l[0];
    for (int i = 1; i < l.size(); i++) {
        if (maior < l[i]) {
            maior = l[i];
        }
    }
    return maior;
}

std::vector<int> Lib::countingSort(std::vector<int> l) {
    if (l.empty())
        return l;
    int k = findMaior(l);
    std::vector<int> lista(k + 1, 0);
    for (int i = 0; i < l.size(); i++) {
        lista[l[i]] += 1;
    }
    for (int i = 1; i < k + 1; i++) {
        lista[i] += lista[i - 1];
    }
    std::vector<int> saida(l.size());
    for (int i = l.size() - 1; i >= 0; i--) {
        int valor = l[i];
        int posicaoSaida = lista[valor] - 1;
        saida[posicaoSaida] = valor;
        lista[valor]--;
    }
    return saida;
}

void Lib::bubble(std::vector<int> &lista, int tam) {
    int temp, flag;
    if (tam) {
        for (int i = 0; i < tam - 1; i++) {
            flag = 0;
            for (int j = 0; j < tam - 1; j++) {
                if (lista[j + 1] < lista[j]) {
                    temp = lista[j];
                    lista[j] = lista[j + 1];
                    lista[j + 1] = temp;
                    flag = 1;
                }
            }
            if (!flag) {
                break;
            }
        }
    }
}

std::vector<int> Lib::bucketSort(std::vector<int> v, int tam) {
    struct Bucket {
        int topo;
        std::vector<int> balde;
    };
    
    Bucket b[10];
    int i, j, k;
    for (i = 0; i < 10; i++)
        b[i].topo = 0;

    for (i = 0; i < tam; i++) {
        j = 9;
        while (1) {
            if (j < 0)
                break;
            if (v[i] >= j * 10) {
                b[j].balde.push_back(v[i]);
                (b[j].topo)++;
                break;
            }
            j--;
        }
    }

    for (i = 0; i < 10; i++)
        if (b[i].topo)
            bubble(b[i].balde, b[i].topo);

    i = 0;
    for (j = 0; j < 10; j++) {
        for (k = 0; k < b[j].topo; k++) {
            v[i] = b[j].balde[k];
            i++;
        }
    }
    return v;
}

std::vector<int> Lib::radixSort(std::vector<int> lista) {
    int maior = lista[0], exp = 1, tamanho = lista.size();
    std::vector<int> auxiliar(tamanho);
    for (int i = 1; i < tamanho; i++) {
        if (lista[i] > maior) {
            maior = lista[i];
        }
    }
    while (maior / exp > 0) {
        std::vector<int> baldes(10, 0);
        for (int i = 0; i < tamanho; i++) {
            baldes[(lista[i] / exp) % 10]++;
        }
        for (int i = 1; i < 10; i++)
            baldes[i] += baldes[i - 1];
        for (int i = tamanho - 1; i >= 0; i--)
            auxiliar[--baldes[(lista[i] / exp) % 10]] = lista[i];
        for (int i = 0; i < tamanho; i++)
            lista[i] = auxiliar[i];
        exp *= 10;
    }
    return lista;
}

// Geometric functions
double Lib::distancia2Pontos(Ponto a, Ponto b) {
    return hypot(a.x - b.x, a.y - b.y);
}

double Lib::distanciaPontoReta(Ponto A, Ponto B, Ponto P) {
    double numerador = fabs((B.x - A.x) * (A.y - P.y) - (A.x - P.x) * (B.y - A.y));
    double denominador = sqrt(pow(B.x - A.x, 2) + pow(B.y - A.y, 2));
    return numerador / denominador;
}

double Lib::areaSecaoTransversal(std::vector<Ponto>& pontos) {
    if (pontos.size() < 3) return 0.0; // Precisa de pelo menos 3 pontos para formar uma área
    
    double area = 0.0;
    int n = pontos.size();
    
    // Fórmula do Shoelace (Teorema de Green)
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        area += (pontos[i].x * pontos[j].y) - (pontos[j].x * pontos[i].y);
    }
    
    return fabs(area) / 2.0;
}

// Math functions
long long Lib::binaryExponecial(int a, int b) {
    if (b == 0) {
        return 1;
    }
    
    if (b % 2 == 0) {
        long long valor = binaryExponecial(a, b/2);
        return valor * valor;
    } else {
        long long valor = binaryExponecial(a, b/2);
        return a * valor * valor;
    }
}

// Graph functions
void Lib::dfs(int v, std::vector<Vertice> &lista, std::vector<bool> &visitado, std::vector<Vertice> &verticesBusca) {
    visitado[v] = true;
    verticesBusca.push_back(lista[v]);

    for (int vizinho : lista[v].arestas) {
        if (!visitado[vizinho]) {
            dfs(vizinho, lista, visitado, verticesBusca);
        }
    }
}

// Greedy Algorithms
double Lib::fractionalKnapsack(std::vector<Item>& items, double capacity) {
    // Sort items by value/weight ratio
    std::sort(items.begin(), items.end(), 
        [](const Item& a, const Item& b) {
            return (a.value / a.weight) > (b.value / b.weight);
        });
    
    double totalValue = 0.0;
    double currentWeight = 0.0;
    
    for (const Item& item : items) {
        if (currentWeight + item.weight <= capacity) {
            currentWeight += item.weight;
            totalValue += item.value;
        } else {
            double remainingWeight = capacity - currentWeight;
            totalValue += item.value * (remainingWeight / item.weight);
            break;
        }
    }
    
    return totalValue;
}

std::vector<int> Lib::coinChange(int amount, std::vector<int>& coins) {
    std::sort(coins.rbegin(), coins.rend()); // Sort in descending order
    std::vector<int> result;
    
    for (int coin : coins) {
        while (amount >= coin) {
            result.push_back(coin);
            amount -= coin;
        }
    }
    
    return result;
}

std::vector<std::pair<int, int>> Lib::taskScheduling(std::vector<std::pair<int, int>>& tasks) {
    // Sort by deadline
    std::sort(tasks.begin(), tasks.end(), 
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
    
    std::vector<std::pair<int, int>> schedule;
    int currentTime = 0;
    
    for (const auto& task : tasks) {
        if (currentTime + task.first <= task.second) {
            schedule.push_back(task);
            currentTime += task.first;
        }
    }
    
    return schedule;
}

// Divide and Conquer
void Lib::mergeSortHelper(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSortHelper(arr, left, mid);
        mergeSortHelper(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

void Lib::merge(std::vector<int>& arr, int left, int mid, int right) {
    std::vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;
    
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    
    for (i = 0; i < k; i++) {
        arr[left + i] = temp[i];
    }
}

std::vector<int> Lib::mergeSort(std::vector<int>& arr) {
    mergeSortHelper(arr, 0, arr.size() - 1);
    return arr;
}

long long Lib::mergeAndCount(std::vector<int>& arr, int left, int mid, int right) {
    std::vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;
    long long inversions = 0;
    
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
            inversions += mid - i + 1;
        }
    }
    
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    
    for (i = 0; i < k; i++) {
        arr[left + i] = temp[i];
    }
    
    return inversions;
}

long long Lib::inversionCount(std::vector<int>& arr) {
    std::vector<int> temp = arr;
    return mergeAndCount(temp, 0, (temp.size() - 1) / 2, temp.size() - 1);
}

std::string Lib::longestCommonPrefix(std::vector<std::string>& strs) {
    if (strs.empty()) return "";
    if (strs.size() == 1) return strs[0];
    
    auto minmax = std::minmax_element(strs.begin(), strs.end());
    const std::string& first = *minmax.first;
    const std::string& last = *minmax.second;
    
    int i = 0;
    while (i < first.length() && i < last.length() && first[i] == last[i]) {
        i++;
    }
    
    return first.substr(0, i);
}

// Graph Algorithms
void Lib::bfsMatrix(std::vector<std::vector<int>>& graph, int start, std::vector<int>& result) {
    int n = graph.size();
    std::vector<bool> visited(n, false);
    std::queue<int> q;
    
    visited[start] = true;
    q.push(start);
    
    while (!q.empty()) {
        int v = q.front();
        q.pop();
        result.push_back(v);
        
        for (int i = 0; i < n; i++) {
            if (graph[v][i] && !visited[i]) {
                visited[i] = true;
                q.push(i);
            }
        }
    }
}

void Lib::dfsMatrix(std::vector<std::vector<int>>& graph, int start, std::vector<int>& result) {
    int n = graph.size();
    std::vector<bool> visited(n, false);
    std::stack<int> s;
    
    s.push(start);
    
    while (!s.empty()) {
        int v = s.top();
        s.pop();
        
        if (!visited[v]) {
            visited[v] = true;
            result.push_back(v);
            
            for (int i = n - 1; i >= 0; i--) {
                if (graph[v][i] && !visited[i]) {
                    s.push(i);
                }
            }
        }
    }
}

void Lib::bfsList(std::vector<Vertice>& graph, int start, std::vector<int>& result) {
    std::vector<bool> visited(graph.size(), false);
    std::queue<int> q;
    
    visited[start] = true;
    q.push(start);
    
    while (!q.empty()) {
        int v = q.front();
        q.pop();
        result.push_back(v);
        
        for (int u : graph[v].arestas) {
            if (!visited[u]) {
                visited[u] = true;
                q.push(u);
            }
        }
    }
}

// Dynamic Programming Algorithms
std::string Lib::longestCommonSubsequence(const std::string& text1, const std::string& text2) {
    int m = text1.length();
    int n = text2.length();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    
    // Fill the dp table
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i-1] == text2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = std::max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    
    // Reconstruct the LCS
    std::string lcs;
    int i = m, j = n;
    while (i > 0 && j > 0) {
        if (text1[i-1] == text2[j-1]) {
            lcs = text1[i-1] + lcs;
            i--; j--;
        } else if (dp[i-1][j] > dp[i][j-1]) {
            i--;
        } else {
            j--;
        }
    }
    
    return lcs;
}

int Lib::knapsack01(const std::vector<int>& weights, const std::vector<int>& values, int capacity) {
    int n = weights.size();
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));
    
    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= capacity; w++) {
            if (weights[i-1] <= w) {
                dp[i][w] = std::max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w]);
            } else {
                dp[i][w] = dp[i-1][w];
            }
        }
    }
    
    return dp[n][capacity];
}

int Lib::matrixChainMultiplication(const std::vector<int>& dimensions) {
    int n = dimensions.size() - 1;
    std::vector<std::vector<int>> dp(n, std::vector<int>(n, 0));
    
    // Length of chain
    for (int len = 2; len <= n; len++) {
        // Starting index of the chain
        for (int i = 0; i < n - len + 1; i++) {
            int j = i + len - 1;
            dp[i][j] = INT_MAX;
            
            // Try all possible splits
            for (int k = i; k < j; k++) {
                int cost = dp[i][k] + dp[k+1][j] + 
                          dimensions[i] * dimensions[k+1] * dimensions[j+1];
                dp[i][j] = std::min(dp[i][j], cost);
            }
        }
    }
    
    return dp[0][n-1];
}

int Lib::longestIncreasingSubsequence(const std::vector<int>& nums) {
    if (nums.empty()) return 0;
    
    std::vector<int> dp(nums.size(), 1);
    int maxLen = 1;
    
    for (int i = 1; i < nums.size(); i++) {
        for (int j = 0; j < i; j++) {
            if (nums[i] > nums[j]) {
                dp[i] = std::max(dp[i], dp[j] + 1);
                maxLen = std::max(maxLen, dp[i]);
            }
        }
    }
    
    return maxLen;
} 