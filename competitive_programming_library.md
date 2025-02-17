# Biblioteca para Maratona de Programação

Esta biblioteca contém implementações completas de algoritmos comumente utilizados em competições de programação. Cada seção contém uma explicação detalhada do algoritmo e sua implementação em C++.

## Índice
1. [Algoritmos de Ordenação](#algoritmos-de-ordenação)
   - [Counting Sort](#counting-sort)
   - [Bucket Sort](#bucket-sort)
   - [Radix Sort](#radix-sort)
   - [Bubble Sort](#bubble-sort)
   - [Merge Sort](#merge-sort)
2. [Algoritmos Gulosos](#algoritmos-gulosos)
   - [Mochila Fracionária](#mochila-fracionária)
   - [Problema do Troco](#problema-do-troco)
   - [Escalonamento de Tarefas](#escalonamento-de-tarefas)
3. [Dividir para Conquistar](#dividir-para-conquistar)
   - [Merge Sort](#merge-sort)
   - [Índice de Inversão](#índice-de-inversão)
   - [Maior Prefixo Comum](#maior-prefixo-comum)
4. [Processamento de Strings](#processamento-de-strings)
   - [Análise de Caracteres](#análise-de-caracteres)
   - [Busca de Padrões](#busca-de-padrões)
   - [Tokenização](#tokenização)
5. [Geometria Computacional](#geometria-computacional)
   - [Distância entre Pontos](#distância-entre-pontos)
   - [Distância Ponto-Reta](#distância-ponto-reta)
   - [Área da Seção Transversal](#área-da-seção-transversal)
6. [Grafos](#grafos)
   - [BFS (Busca em Largura)](#bfs)
   - [DFS (Busca em Profundidade)](#dfs)
7. [Matemática](#matemática)
   - [Exponenciação Binária](#exponenciação-binária)
8. [Programação Dinâmica](#programação-dinâmica)
   - [Maior Subsequência Comum](#maior-subsequência-comum)
   - [Mochila 0/1](#mochila-01)
   - [Multiplicação de Cadeia de Matrizes](#multiplicação-de-cadeia-de-matrizes)
   - [Maior Subsequência Crescente](#maior-subsequência-crescente)

## Algoritmos de Ordenação

### Counting Sort
**Complexidade**: O(n + k), onde k é o maior elemento
**Melhor uso**: Arrays com números inteiros pequenos e muitas repetições

```cpp
// Função auxiliar para encontrar o maior elemento
int findMaior(vector<int> l) {
    int maior = l[0];
    for (int i = 1; i < l.size(); i++) {
        if (maior < l[i]) {
            maior = l[i];
        }
    }
    return maior;
}

vector<int> countingSort(vector<int> l) {
    if (l.empty())
        return l;
    int k = findMaior(l);
    vector<int> lista(k + 1, 0);
    for (int i = 0; i < l.size(); i++) {
        lista[l[i]] += 1;
    }
    for (int i = 1; i < k + 1; i++) {
        lista[i] += lista[i - 1];
    }
    vector<int> saida(l.size());
    for (int i = l.size() - 1; i >= 0; i--) {
        int valor = l[i];
        int posicaoSaida = lista[valor] - 1;
        saida[posicaoSaida] = valor;
        lista[valor]--;
    }
    return saida;
}
```

**Quando usar**: 
- Quando os elementos são inteiros conhecidos
- Quando o intervalo de valores não é muito maior que o número de elementos
- Quando você precisa de ordenação estável

### Bubble Sort
**Complexidade**: O(n²)
**Melhor uso**: Arrays pequenos ou quase ordenados

```cpp
void bubble(vector<int> &lista, int tam) {
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
```

### Bucket Sort
**Complexidade**: O(n + k), onde k é o número de baldes
**Melhor uso**: Distribuição uniforme dos dados

```cpp
vector<int> bucketSort(vector<int> v, int tam) {
    struct Bucket {
        int topo;
        vector<int> balde;
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
```

### Radix Sort
**Complexidade**: O(d * (n + k)), onde d é o número de dígitos
**Melhor uso**: Números inteiros com quantidade fixa de dígitos

```cpp
vector<int> radixSort(vector<int> lista) {
    int maior = lista[0], exp = 1, tamanho = lista.size();
    vector<int> auxiliar(tamanho);
    
    for (int i = 1; i < tamanho; i++) {
        if (lista[i] > maior) {
            maior = lista[i];
        }
    }
    
    while (maior / exp > 0) {
        vector<int> baldes(10, 0);
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
```

### Merge Sort
**Complexidade**: O(n log n)
**Melhor uso**: Ordenação estável e garantida

```cpp
void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);
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

void mergeSortHelper(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSortHelper(arr, left, mid);
        mergeSortHelper(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

vector<int> mergeSort(vector<int>& arr) {
    mergeSortHelper(arr, 0, arr.size() - 1);
    return arr;
}
```

## Algoritmos Gulosos

### Mochila Fracionária
**Complexidade**: O(n log n)
**Melhor uso**: Quando é possível fracionar os itens

```cpp
struct Item {
    double weight;
    double value;
};

double fractionalKnapsack(vector<Item>& items, double capacity) {
    // Sort items by value/weight ratio
    sort(items.begin(), items.end(), 
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
```

### Problema do Troco
**Complexidade**: O(n)
**Melhor uso**: Quando a moeda de maior valor é sempre a melhor escolha

```cpp
vector<int> coinChange(int amount, vector<int>& coins) {
    sort(coins.rbegin(), coins.rend()); // Sort in descending order
    vector<int> result;
    
    for (int coin : coins) {
        while (amount >= coin) {
            result.push_back(coin);
            amount -= coin;
        }
    }
    
    return result;
}
```

### Escalonamento de Tarefas
**Complexidade**: O(n log n)
**Melhor uso**: Quando as tarefas têm deadlines e durações fixas

```cpp
vector<pair<int, int>> taskScheduling(vector<pair<int, int>>& tasks) {
    // Sort by deadline
    sort(tasks.begin(), tasks.end(), 
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
    
    vector<pair<int, int>> schedule;
    int currentTime = 0;
    
    for (const auto& task : tasks) {
        if (currentTime + task.first <= task.second) {
            schedule.push_back(task);
            currentTime += task.first;
        }
    }
    
    return schedule;
}
```

## Dividir para Conquistar

### Merge Sort
**Complexidade**: O(n log n)
**Melhor uso**: Ordenação estável e garantida

```cpp
void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);
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

void mergeSortHelper(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSortHelper(arr, left, mid);
        mergeSortHelper(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

vector<int> mergeSort(vector<int>& arr) {
    mergeSortHelper(arr, 0, arr.size() - 1);
    return arr;
}
```

### Índice de Inversão
**Complexidade**: O(n log n)
**Melhor uso**: Contar quantas inversões são necessárias para ordenar um array

```cpp
long long mergeAndCount(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);
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

long long inversionCount(vector<int>& arr) {
    vector<int> temp = arr;
    return mergeAndCount(temp, 0, (temp.size() - 1) / 2, temp.size() - 1);
}
```

### Maior Prefixo Comum
**Complexidade**: O(n * m), onde n é o número de strings e m é o comprimento da menor string
**Melhor uso**: Encontrar o maior prefixo comum entre várias strings

```cpp
string longestCommonPrefix(vector<string>& strs) {
    if (strs.empty()) return "";
    if (strs.size() == 1) return strs[0];
    
    auto minmax = minmax_element(strs.begin(), strs.end());
    const string& first = *minmax.first;
    const string& last = *minmax.second;
    
    int i = 0;
    while (i < first.length() && i < last.length() && first[i] == last[i]) {
        i++;
    }
    
    return first.substr(0, i);
}
```

## Processamento de Strings

### Análise de Caracteres
**Funcionalidade**: Conta vogais, consoantes e dígitos em uma string

```cpp
struct CharacterAnalysis {
    int digits;
    int vowels;
    int consonants;
};

bool isVowel(char c) {
    c = tolower(c);
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
}

bool isConsonant(char c) {
    return isalpha(c) && !isVowel(c);
}

CharacterAnalysis analyzeCharacters(const string& T) {
    CharacterAnalysis analysis = {0, 0, 0};
    
    for (char c : T) {
        if (isdigit(c)) analysis.digits++;
        else if (isVowel(c)) analysis.vowels++;
        else if (isConsonant(c)) analysis.consonants++;
    }
    
    return analysis;
}

string toLowerCase(const string& T) {
    string result = T;
    transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}
```

### Busca de Padrões
**Complexidade**: O(n*m), onde n é o tamanho do texto e m do padrão

```cpp
vector<int> findAllOccurrences(const string& T, const string& P) {
    vector<int> positions;
    size_t pos = T.find(P);
    
    while (pos != string::npos) {
        positions.push_back(pos);
        pos = T.find(P, pos + 1);
    }
    
    return positions.empty() ? vector<int>{-1} : positions;
}
```

### Tokenização
**Complexidade**: O(n)

```cpp
vector<string> tokenize(const string& T) {
    vector<string> tokens;
    stringstream ss(toLowerCase(T));
    string token;
    
    while (ss >> token) {
        token.erase(remove(token.begin(), token.end(), '.'), token.end());
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    
    sort(tokens.begin(), tokens.end());
    return tokens;
}

string findSmallestToken(const string& T) {
    auto tokens = tokenize(T);
    return tokens.empty() ? "" : tokens[0];
}

vector<string> findMostFrequentWords(const string& T) {
    auto tokens = tokenize(T);
    map<string, int> frequency;
    int maxFreq = 0;
    
    for (const auto& token : tokens) {
        maxFreq = max(maxFreq, ++frequency[token]);
    }
    
    vector<string> mostFrequent;
    for (const auto& pair : frequency) {
        if (pair.second == maxFreq) {
            mostFrequent.push_back(pair.first);
        }
    }
    
    return mostFrequent;
}
```

## Geometria Computacional

### Estruturas Básicas
```cpp
struct Ponto {
    double x;
    double y;
};
```

### Distância entre Pontos
**Complexidade**: O(1)

```cpp
double distancia2Pontos(Ponto a, Ponto b) {
    return hypot(a.x - b.x, a.y - b.y);
}
```

### Distância Ponto-Reta
**Complexidade**: O(1)

```cpp
double distanciaPontoReta(Ponto A, Ponto B, Ponto P) {
    double numerador = fabs((B.x - A.x) * (A.y - P.y) - (A.x - P.x) * (B.y - A.y));
    double denominador = sqrt(pow(B.x - A.x, 2) + pow(B.y - A.y, 2));
    return numerador / denominador;
}
```

### Área da Seção Transversal
**Complexidade**: O(n), onde n é o número de pontos
**Melhor uso**: Calcular a área de um polígono definido por uma sequência de pontos

```cpp
double areaSecaoTransversal(vector<Ponto>& pontos) {
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
```

A função `areaSecaoTransversal` utiliza a Fórmula do Shoelace (também conhecida como Fórmula da Área de Gauss ou Teorema de Green) para calcular a área de um polígono dado seus vértices. Os pontos devem ser fornecidos em ordem (horária ou anti-horária). O algoritmo funciona para qualquer polígono simples (sem auto-interseções).

**Observações importantes**:
- Os pontos devem estar ordenados formando o perímetro do polígono
- Funciona tanto para polígonos convexos quanto côncavos
- O resultado é sempre positivo (usa-se fabs para garantir isso)
- Retorna 0 se houver menos de 3 pontos (impossível formar área)

## Grafos

### Estruturas Básicas
```cpp
struct Vertice {
    int vertice;
    vector<int> arestas;
};
```

### BFS (Busca em Largura)
**Complexidade**: O(V + E) para lista de adjacência, O(V²) para matriz de adjacência
**Melhor uso**: Encontrar caminhos mais curtos em grafos não ponderados

```cpp
// Matriz de Adjacência
void bfsMatrix(vector<vector<int>>& graph, int start, vector<int>& result) {
    int n = graph.size();
    vector<bool> visited(n, false);
    queue<int> q;
    
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

// Lista de Adjacência
void bfsList(vector<Vertice>& graph, int start, vector<int>& result) {
    vector<bool> visited(graph.size(), false);
    queue<int> q;
    
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
```

### DFS (Busca em Profundidade)
**Complexidade**: O(V + E) para lista de adjacência, O(V²) para matriz de adjacência
**Melhor uso**: Explorar todas as possibilidades em um grafo

```cpp
// Matriz de Adjacência
void dfsMatrix(vector<vector<int>>& graph, int start, vector<int>& result) {
    int n = graph.size();
    vector<bool> visited(n, false);
    stack<int> s;
    
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
```

## Matemática

### Exponenciação Binária
**Complexidade**: O(log n)

```cpp
long long binaryExponecial(int a, int b) {
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
```

## Programação Dinâmica

### Maior Subsequência Comum
**Complexidade**: O(mn), onde m e n são os comprimentos das strings
**Melhor uso**: Quando precisamos encontrar a maior sequência de caracteres que aparece em duas strings na mesma ordem

```cpp
string longestCommonSubsequence(const string& text1, const string& text2) {
    int m = text1.length();
    int n = text2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
    
    // Preenche a tabela dp
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i-1] == text2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    
    // Reconstrói a subsequência
    string lcs;
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
```

**Quando usar**:
- Comparação de strings
- Análise de DNA/RNA
- Detecção de plágio
- Diff de arquivos

### Mochila 0/1
**Complexidade**: O(nW), onde n é o número de itens e W é a capacidade
**Melhor uso**: Quando os itens não podem ser fracionados

```cpp
int knapsack01(const vector<int>& weights, const vector<int>& values, int capacity) {
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(capacity + 1, 0));
    
    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= capacity; w++) {
            if (weights[i-1] <= w) {
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w]);
            } else {
                dp[i][w] = dp[i-1][w];
            }
        }
    }
    
    return dp[n][capacity];
}
```

**Quando usar**:
- Problemas de otimização com restrições
- Seleção de itens com peso e valor
- Alocação de recursos limitados

### Multiplicação de Cadeia de Matrizes
**Complexidade**: O(n³), onde n é o número de matrizes
**Melhor uso**: Otimização de operações de multiplicação de matrizes

```cpp
int matrixChainMultiplication(const vector<int>& dimensions) {
    int n = dimensions.size() - 1;
    vector<vector<int>> dp(n, vector<int>(n, 0));
    
    for (int len = 2; len <= n; len++) {
        for (int i = 0; i < n - len + 1; i++) {
            int j = i + len - 1;
            dp[i][j] = INT_MAX;
            
            for (int k = i; k < j; k++) {
                int cost = dp[i][k] + dp[k+1][j] + 
                          dimensions[i] * dimensions[k+1] * dimensions[j+1];
                dp[i][j] = min(dp[i][j], cost);
            }
        }
    }
    
    return dp[0][n-1];
}
```

**Quando usar**:
- Otimização de multiplicação de matrizes
- Problemas de parentização
- Minimização de operações

### Maior Subsequência Crescente
**Complexidade**: O(n²)
**Melhor uso**: Encontrar a maior sequência de números que está em ordem crescente

```cpp
int longestIncreasingSubsequence(const vector<int>& nums) {
    if (nums.empty()) return 0;
    
    vector<int> dp(nums.size(), 1);
    int maxLen = 1;
    
    for (int i = 1; i < nums.size(); i++) {
        for (int j = 0; j < i; j++) {
            if (nums[i] > nums[j]) {
                dp[i] = max(dp[i], dp[j] + 1);
                maxLen = max(maxLen, dp[i]);
            }
        }
    }
    
    return maxLen;
}
```

**Quando usar**:
- Análise de sequências
- Problemas de otimização de sequências
- Detecção de padrões crescentes em dados

## Headers Necessários
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <cmath>
#include <sstream>
#include <cctype>
using namespace std;
```

## Dicas de Uso em Competição

1. **Escolha do Algoritmo de Ordenação**:
   - Para n < 100: Qualquer algoritmo serve
   - Para números pequenos (0-1000): Counting Sort
   - Para dados uniformemente distribuídos: Bucket Sort
   - Para números com dígitos fixos: Radix Sort

2. **Processamento de Strings**:
   - Use `toLowerCase()` antes de comparações
   - `findAllOccurrences()` para busca de padrões
   - `analyzeCharacters()` para análise rápida de texto

3. **Geometria**:
   - Sempre use `double` para coordenadas
   - Cuidado com precisão em comparações (use EPS = 1e-9)
   - Use `distanciaPontoReta()` para problemas de posição relativa

4. **Grafos**:
   - DFS é mais simples de implementar que BFS
   - Mantenha um vetor de visitados
   - Use recursão com cuidado (limite da pilha)
   - Para grafos grandes, considere usar lista de adjacência

5. **Matemática**:
   - Use exponenciação binária para potências grandes
   - Cuidado com overflow em operações matemáticas
   - Para MOD, use: `(a * b) % MOD` após cada operação

6. **Otimizações de Input/Output**:
```cpp
// Adicione no início do main
ios_base::sync_with_stdio(false);
cin.tie(NULL);
```

7. **Constantes Úteis**:
```cpp
const double PI = acos(-1.0);
const double EPS = 1e-9;
const int MOD = 1e9 + 7;
const int INF = 0x3f3f3f3f;
``` 