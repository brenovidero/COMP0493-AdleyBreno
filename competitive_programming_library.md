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
   - [KMP](#kmp)
   - [Boyer Moore](#boyer-moore)
   - [Rabin-Karp](#rabin-karp)
5. [Geometria Computacional](#geometria-computacional)
   - [Distância entre Pontos](#distância-entre-pontos)
   - [Distância Ponto-Reta](#distância-ponto-reta)
   - [Área da Seção Transversal](#área-da-seção-transversal)
6. [Grafos](#grafos)
   - [BFS (Busca em Largura)](#bfs)
   - [DFS (Busca em Profundidade)](#dfs)
   - [Ford-Fulkerson](#ford-fulkerson)
   - [Edmonds-Karp](#edmonds-karp)
   - [Dinic](#dinic)
   - [Union-Find](#union-find)
   - [Kruskal](#kruskal)
   - [Prim](#prim)
7. [Matemática](#matemática)
   - [Exponenciação Binária](#exponenciação-binária)
   - [Coeficiente Binomial](#coeficiente-binomial)
   - [Teste de Primalidade](#teste-de-primalidade)
   - [Inverso Modular](#inverso-modular)
8. [Programação Dinâmica](#programação-dinâmica)
   - [Maior Subsequência Comum](#maior-subsequência-comum)
   - [Mochila 0/1](#mochila-01)
   - [Multiplicação de Cadeia de Matrizes](#multiplicação-de-cadeia-de-matrizes)
   - [Maior Subsequência Crescente](#maior-subsequência-crescente)
9. [Estruturas de Dados](#estruturas-de-dados)
   - [Fenwick Tree](#fenwick-tree)
   - [Segment Tree](#segment-tree)

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

### KMP
**Complexidade**: O(n + m), onde n é o tamanho do texto e m é o tamanho do padrão
**Melhor uso**: Busca eficiente de padrões em texto

O algoritmo Knuth-Morris-Pratt (KMP) é um algoritmo de busca de padrões que utiliza a informação de correspondências anteriores para evitar comparações redundantes.

```cpp
vector<int> computeLPS(const string& pattern) {
    int m = pattern.length();
    vector<int> lps(m, 0);
    
    int len = 0;
    int i = 1;
    
    while (i < m) {
        if (pattern[i] == pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
    
    return lps;
}

vector<int> kmp(const string& text, const string& pattern) {
    int n = text.length();
    int m = pattern.length();
    vector<int> matches;
    
    if (m == 0 || m > n)
        return matches;
    
    // Preprocess pattern to get longest prefix suffix array
    vector<int> lps = computeLPS(pattern);
    
    int i = 0; // Index for text[]
    int j = 0; // Index for pattern[]
    
    while (i < n) {
        if (pattern[j] == text[i]) {
            i++;
            j++;
        }
        
        if (j == m) {
            matches.push_back(i - j); // Found a match
            j = lps[j - 1];
        } else if (i < n && pattern[j] != text[i]) {
            if (j != 0)
                j = lps[j - 1];
            else
                i++;
        }
    }
    
    return matches;
}
```

**Quando usar**:
- Busca de padrões em textos
- Quando o padrão pode se repetir dentro de si mesmo
- Quando você precisa encontrar todas as ocorrências de um padrão no texto

### Boyer-Moore
**Complexidade**: O(n + m) na média, O(n*m) no pior caso
**Melhor uso**: Busca eficiente de padrões em textos grandes, especialmente com alfabetos grandes

O algoritmo Boyer-Moore é considerado o mais eficiente algoritmo de busca de padrões em prática para linguagens naturais. Ele usa duas heurísticas: "bad character" e "good suffix".

```cpp
vector<int> buildBadCharTable(const string& pattern) {
    int m = pattern.length();
    vector<int> badChar(256, -1);
    
    for (int i = 0; i < m; i++)
        badChar[pattern[i]] = i;
    
    return badChar;
}

vector<int> buildGoodSuffixTable(const string& pattern) {
    int m = pattern.length();
    vector<int> shift(m, 0);
    vector<int> border(m, 0);
    
    // Preprocessing for case 2
    int j = m;
    border[m - 1] = j;
    
    for (int i = m - 2; i >= 0; i--) {
        while (j < m && pattern[i] != pattern[j - 1])
            j = border[j];
        
        j--;
        border[i] = j;
    }
    
    // Preprocessing for case 1
    for (int i = 0; i < m; i++)
        shift[i] = m;
    
    j = 0;
    for (int i = m - 1; i >= 0; i--) {
        if (border[i] == i + 1) {
            while (j < m - 1 - i)
                shift[j++] = m - 1 - i;
        }
    }
    
    for (int i = 0; i <= m - 2; i++)
        shift[m - 1 - border[i]] = m - 1 - i;
    
    return shift;
}

vector<int> boyerMoore(const string& text, const string& pattern) {
    int n = text.length();
    int m = pattern.length();
    vector<int> matches;
    
    if (m == 0 || m > n)
        return matches;
    
    // Preprocess pattern
    vector<int> badChar = buildBadCharTable(pattern);
    vector<int> goodSuffix = buildGoodSuffixTable(pattern);
    
    int s = 0; // Shift of the pattern relative to text
    
    while (s <= n - m) {
        int j = m - 1;
        
        // Match pattern from right to left
        while (j >= 0 && pattern[j] == text[s + j])
            j--;
        
        if (j < 0) {
            matches.push_back(s); // Pattern found at position s
            s += (s + m < n) ? m - badChar[text[s + m]] : 1;
        } else {
            // Bad Character heuristic
            int badCharShift = j - badChar[text[s + j]];
            if (badCharShift < 1) badCharShift = 1;
            
            // Good Suffix heuristic
            int goodSuffixShift = goodSuffix[j];
            
            s += max(badCharShift, goodSuffixShift);
        }
    }
    
    return matches;
}
```

**Quando usar**:
- Busca de padrões em textos grandes
- Quando o alfabeto (conjunto de caracteres) é grande
- Quando o padrão não muda frequentemente (pré-processamento mais pesado)

### Rabin-Karp
**Complexidade**: O(n + m) em média, O(n*m) no pior caso
**Melhor uso**: Busca de múltiplos padrões em um texto

O algoritmo Rabin-Karp utiliza hashing para encontrar um padrão em um texto. Ele calcula um hash do padrão e compara com os hashes de substrings do texto.

```cpp
vector<int> rabinKarp(const string& text, const string& pattern) {
    int n = text.length();
    int m = pattern.length();
    vector<int> matches;
    
    if (m == 0 || m > n)
        return matches;
    
    const int prime = 101; // A prime number
    const int d = 256;     // Number of characters in the alphabet
    
    // Calculate hash for pattern and first window of text
    int patternHash = 0;
    int textHash = 0;
    int h = 1;
    
    // Calculate h = pow(d, m-1) % prime
    for (int i = 0; i < m - 1; i++)
        h = (h * d) % prime;
    
    // Calculate hash value for pattern and first window of text
    for (int i = 0; i < m; i++) {
        patternHash = (d * patternHash + pattern[i]) % prime;
        textHash = (d * textHash + text[i]) % prime;
    }
    
    // Slide the pattern over text one by one
    for (int i = 0; i <= n - m; i++) {
        // Check if the hash values match
        if (patternHash == textHash) {
            // Check characters one by one
            bool match = true;
            for (int j = 0; j < m; j++) {
                if (text[i + j] != pattern[j]) {
                    match = false;
                    break;
                }
            }
            
            if (match)
                matches.push_back(i);
        }
        
        // Calculate hash value for next window of text
        if (i < n - m) {
            textHash = (d * (textHash - text[i] * h) + text[i + m]) % prime;
            
            // We might get negative hash, convert it to positive
            if (textHash < 0)
                textHash += prime;
        }
    }
    
    return matches;
}
```

**Quando usar**:
- Busca de padrões em textos
- Busca de múltiplos padrões no mesmo texto (com pequenas modificações)
- Detecção de plágio ou busca de substrings

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

### Ford-Fulkerson
**Complexidade**: O(E * max_flow), onde E é o número de arestas
**Melhor uso**: Problemas de fluxo máximo em grafos

O algoritmo de Ford-Fulkerson é utilizado para encontrar o fluxo máximo em uma rede de fluxo. Ele itera aumentando o fluxo ao longo dos caminhos do vértice fonte ao vértice sumidouro até que não haja mais caminhos possíveis.

```cpp
bool bfs(vector<vector<int>>& residualGraph, int s, int t, vector<int>& parent) {
    int n = residualGraph.size();
    vector<bool> visited(n, false);
    queue<int> q;
    
    q.push(s);
    visited[s] = true;
    parent[s] = -1;
    
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        
        for (int v = 0; v < n; v++) {
            if (!visited[v] && residualGraph[u][v] > 0) {
                if (v == t) {
                    parent[v] = u;
                    return true;
                }
                q.push(v);
                parent[v] = u;
                visited[v] = true;
            }
        }
    }
    return false;
}

int fordFulkerson(vector<vector<int>>& graph, int source, int sink) {
    int n = graph.size();
    vector<vector<int>> residualGraph = graph;
    vector<int> parent(n);
    int maxFlow = 0;
    
    while (bfs(residualGraph, source, sink, parent)) {
        int pathFlow = INT_MAX;
        for (int v = sink; v != source; v = parent[v]) {
            int u = parent[v];
            pathFlow = min(pathFlow, residualGraph[u][v]);
        }
        
        for (int v = sink; v != source; v = parent[v]) {
            int u = parent[v];
            residualGraph[u][v] -= pathFlow;
            residualGraph[v][u] += pathFlow;
        }
        
        maxFlow += pathFlow;
    }
    
    return maxFlow;
}
```

**Quando usar**:
- Problemas de fluxo máximo em redes
- Quando os valores de capacidade são pequenos
- Problemas de bipartite matching

### Edmonds-Karp
**Complexidade**: O(V * E²), onde V é o número de vértices e E é o número de arestas
**Melhor uso**: Problemas de fluxo máximo em grafos com garantia de complexidade melhor

O algoritmo de Edmonds-Karp é uma implementação específica do algoritmo de Ford-Fulkerson, que usa BFS para encontrar caminhos aumentantes. Isso garante que o caminho encontrado seja o mais curto possível em termos de número de arestas.

```cpp
// Edmonds-Karp usa a mesma implementação de Ford-Fulkerson, mas com BFS garantido
int edmondsKarp(vector<vector<int>>& graph, int source, int sink) {
    // A implementação já usa BFS na função auxiliar
    return fordFulkerson(graph, source, sink);
}
```

**Quando usar**:
- Problemas de fluxo máximo que exigem melhor garantia de complexidade
- Quando os valores de capacidade são intermediários

### Dinic
**Complexidade**: O(V² * E), onde V é o número de vértices e E é o número de arestas
**Melhor uso**: Problemas de fluxo máximo em grafos com muitas arestas

O algoritmo de Dinic é uma otimização dos algoritmos anteriores, usando BFS para construir níveis de grafo e DFS para encontrar múltiplos caminhos aumentantes de uma vez.

```cpp
bool dinicBfs(const vector<vector<int>>& residualGraph, vector<int>& level, int s, int t) {
    int n = residualGraph.size();
    fill(level.begin(), level.end(), -1);
    level[s] = 0;
    
    queue<int> q;
    q.push(s);
    
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        
        for (int v = 0; v < n; v++) {
            if (level[v] < 0 && residualGraph[u][v] > 0) {
                level[v] = level[u] + 1;
                q.push(v);
            }
        }
    }
    
    return level[t] >= 0;
}

int dinicDfs(vector<vector<int>>& residualGraph, vector<int>& level, 
             vector<int>& ptr, int u, int t, int flow) {
    if (u == t)
        return flow;
    
    int n = residualGraph.size();
    for (int& i = ptr[u]; i < n; i++) {
        int v = i;
        if (level[v] == level[u] + 1 && residualGraph[u][v] > 0) {
            int curr_flow = min(flow, residualGraph[u][v]);
            int temp_flow = dinicDfs(residualGraph, level, ptr, v, t, curr_flow);
            
            if (temp_flow > 0) {
                residualGraph[u][v] -= temp_flow;
                residualGraph[v][u] += temp_flow;
                return temp_flow;
            }
        }
    }
    
    return 0;
}

int dinic(vector<vector<int>>& graph, int source, int sink) {
    int n = graph.size();
    vector<vector<int>> residualGraph = graph;
    vector<int> level(n);
    vector<int> ptr(n);
    int maxFlow = 0;
    
    while (dinicBfs(residualGraph, level, source, sink)) {
        fill(ptr.begin(), ptr.end(), 0);
        while (int flow = dinicDfs(residualGraph, level, ptr, source, sink, INT_MAX))
            maxFlow += flow;
    }
    
    return maxFlow;
}
```

**Quando usar**:
- Problemas de fluxo máximo de grande escala
- Quando você precisa de um algoritmo mais eficiente para casos de teste grandes
- Competições que exigem soluções otimizadas para problemas de fluxo

### Union-Find
**Complexidade**: O(α(n)) por operação, onde α é a função inversa de Ackermann (quase constante)
**Melhor uso**: Detectar ciclos em grafos não-direcionados e operações de união de conjuntos

Union-Find (ou Disjoint Set) é uma estrutura de dados que mantém uma coleção de conjuntos disjuntos e provê operações para unir conjuntos e encontrar o conjunto ao qual um elemento pertence.

```cpp
struct DisjointSet {
    vector<int> parent, rank;
    
    DisjointSet(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; i++)
            parent[i] = i;
    }
    
    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);
        return parent[x];
    }
    
    void unionSets(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        
        if (rootX == rootY)
            return;
        
        if (rank[rootX] < rank[rootY])
            parent[rootX] = rootY;
        else if (rank[rootX] > rank[rootY])
            parent[rootY] = rootX;
        else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
    }
};
```

**Quando usar**:
- Para verificar se dois nós estão no mesmo componente conectado
- Para unir componentes conectados
- Em algoritmos como Kruskal para árvore geradora mínima

### Kruskal
**Complexidade**: O(E log E), onde E é o número de arestas
**Melhor uso**: Encontrar a árvore geradora mínima de um grafo conectado

O algoritmo de Kruskal encontra a árvore geradora mínima (ou máxima) para um grafo conectado com pesos. Ele ordena todas as arestas e vai adicionando-as à MST se não formarem ciclos.

```cpp
vector<pair<int, pair<int, int>>> kruskal(
    vector<pair<int, pair<int, int>>>& edges, int vertices) {
    
    sort(edges.begin(), edges.end()); // Sort edges by weight
    DisjointSet ds(vertices);
    vector<pair<int, pair<int, int>>> result;
    
    for (auto& edge : edges) {
        int weight = edge.first;
        int u = edge.second.first;
        int v = edge.second.second;
        
        if (ds.find(u) != ds.find(v)) {
            result.push_back(edge);
            ds.unionSets(u, v);
        }
    }
    
    return result;
}
```

**Quando usar**:
- Encontrar a árvore geradora mínima de um grafo
- Quando o grafo é esparso (poucas arestas)
- Para otimizar a construção de redes de custo mínimo

### Prim
**Complexidade**: O(E log V) com fila de prioridade, onde E é o número de arestas e V é o número de vértices
**Melhor uso**: Encontrar a árvore geradora mínima de um grafo conectado denso

O algoritmo de Prim começa com um único vértice e cresce a árvore geradora mínima adicionando o vértice mais próximo dos já incluídos.

```cpp
vector<pair<int, int>> prim(vector<vector<pair<int, int>>>& graph, int start) {
    int n = graph.size();
    vector<bool> visited(n, false);
    vector<int> key(n, INT_MAX);
    vector<int> parent(n, -1);
    
    // Use priority queue to find minimum weight edge
    priority_queue<pair<int, int>, vector<pair<int, int>>, 
                  greater<pair<int, int>>> pq;
    
    // Start with vertex 'start'
    key[start] = 0;
    pq.push({0, start});
    
    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        
        if (visited[u])
            continue;
        
        visited[u] = true;
        
        for (auto& neighbor : graph[u]) {
            int v = neighbor.first;
            int weight = neighbor.second;
            
            if (!visited[v] && weight < key[v]) {
                key[v] = weight;
                parent[v] = u;
                pq.push({key[v], v});
            }
        }
    }
    
    // Construct the MST edges
    vector<pair<int, int>> result;
    for (int i = 0; i < n; i++) {
        if (i != start && parent[i] != -1) {
            result.push_back({parent[i], i});
        }
    }
    
    return result;
}
```

**Quando usar**:
- Encontrar a árvore geradora mínima de um grafo
- Quando o grafo é denso (muitas arestas)
- Quando você já tem uma representação de lista de adjacência do grafo

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

### Teste de Primalidade
**Complexidade**: 
- Força bruta: O(sqrt(n))
- Otimizado: O(sqrt(n)) com otimizações

O teste de primalidade verifica se um número é primo, ou seja, se é divisível apenas por 1 e por ele mesmo.

#### Teste de Primalidade (Força Bruta)
```cpp
bool isPrimeNaive(int n) {
    if (n <= 1)
        return false;
    if (n <= 3)
        return true;
    if (n % 2 == 0 || n % 3 == 0)
        return false;
    
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    }
    
    return true;
}
```

#### Teste de Primalidade (Otimizado)
```cpp
bool isPrimeOptimized(int n) {
    if (n <= 1)
        return false;
    if (n <= 3)
        return true;
    if (n % 2 == 0 || n % 3 == 0)
        return false;
    
    // Check using 6k ± 1 optimization
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    }
    
    return true;
}
```

**Quando usar**:
- Verificar se um número é primo
- Como parte de algoritmos de fatoração
- Em problemas de teoria dos números

### Inverso Modular
**Complexidade**: O(log m), onde m é o módulo
**Melhor uso**: Calcular divisões modulares a^(-1) % m

O inverso modular de a módulo m é um número x tal que a*x ≡ 1 (mod m). Ele é usado para calcular divisões em aritmética modular.

```cpp
long long modInverse(long long a, long long m) {
    long long m0 = m;
    long long y = 0, x = 1;
    
    if (m == 1)
        return 0;
    
    while (a > 1) {
        long long q = a / m;
        long long t = m;
        
        m = a % m;
        a = t;
        t = y;
        
        y = x - q * y;
        x = t;
    }
    
    if (x < 0)
        x += m0;
    
    return x;
}
```

**Quando usar**:
- Para calcular (a / b) % m como (a * modInverse(b, m)) % m
- Em algoritmos de combinatória com aritmética modular
- Quando é necessário dividir valores em campos finitos

### Coeficiente Binomial
**Complexidade**: 
- Implementação analítica: O(k)
- Implementação DP: O(n*k)
**Melhor uso**: Calcular combinações, coeficientes binomiais (n escolhe k)

O coeficiente binomial C(n,k) representa o número de maneiras de escolher k elementos de um conjunto de n elementos.

#### Implementação Analítica
```cpp
long long binomialCoefficient(int n, int k) {
    if (k < 0 || k > n)
        return 0;
    if (k == 0 || k == n)
        return 1;
    
    // C(n, k) = C(n, n-k)
    if (k > n - k)
        k = n - k;
    
    long long res = 1;
    
    // Calculate [n * (n-1) * ... * (n-k+1)] / [k * (k-1) * ... * 1]
    for (int i = 0; i < k; ++i) {
        res *= (n - i);
        res /= (i + 1);
    }
    
    return res;
}
```

#### Implementação com Programação Dinâmica
```cpp
long long binomialCoefficientDP(int n, int k) {
    vector<vector<long long>> C(n + 1, vector<long long>(k + 1, 0));
    
    // Base cases
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= min(i, k); j++) {
            if (j == 0 || j == i)
                C[i][j] = 1;
            else
                C[i][j] = C[i - 1][j - 1] + C[i - 1][j];
        }
    }
    
    return C[n][k];
}
```

**Quando usar**:
- Problemas de contagem de combinações
- Expansão binomial
- Problemas de probabilidade
- A implementação DP é preferível para valores grandes de n e k, especialmente ao lidar com números modulares

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

## Estruturas de Dados

### Fenwick Tree
**Complexidade**: 
- Construção: O(n log n)
- Consulta e atualização: O(log n)
**Melhor uso**: Calcular somas prefixas dinâmicas eficientemente

A Fenwick Tree (ou Binary Indexed Tree) é uma estrutura de dados que suporta atualizações de elementos e cálculo de somas prefixas em tempo logarítmico.

```cpp
struct FenwickTree {
    vector<int> bit;
    int size;
    
    FenwickTree(int n) {
        size = n;
        bit.assign(n + 1, 0);
    }
    
    FenwickTree(const vector<int>& arr) {
        size = arr.size();
        bit.assign(size + 1, 0);
        
        for (int i = 0; i < size; i++)
            update(i, arr[i]);
    }
    
    void update(int idx, int val) {
        idx++; // 1-based indexing
        while (idx <= size) {
            bit[idx] += val;
            idx += idx & -idx; // Add LSB
        }
    }
    
    int query(int idx) {
        idx++; // 1-based indexing
        int sum = 0;
        while (idx > 0) {
            sum += bit[idx];
            idx -= idx & -idx; // Remove LSB
        }
        return sum;
    }
    
    int rangeQuery(int l, int r) {
        return query(r) - query(l - 1);
    }
};
```

**Quando usar**:
- Problemas que envolvem somas prefixas com atualizações
- Contagem de inversões
- Consultas de frequência em um intervalo
- Consultas de soma em um intervalo

### Segment Tree
**Complexidade**: 
- Construção: O(n)
- Consulta e atualização pontual: O(log n)
- Atualização em intervalo: O(log n)
**Melhor uso**: Operações em intervalos (range queries)

A Segment Tree é uma estrutura de dados versátil para operações em intervalos, como soma, mínimo, máximo, etc. Ela também suporta atualizações em intervalos com lazy propagation.

```cpp
struct SegmentTree {
    vector<int> tree;
    vector<int> lazy;
    int size;
    
    SegmentTree(int n) {
        size = n;
        tree.assign(4 * n, 0);
        lazy.assign(4 * n, 0);
    }
    
    SegmentTree(const vector<int>& arr) {
        size = arr.size();
        tree.assign(4 * size, 0);
        lazy.assign(4 * size, 0);
        build(arr, 1, 0, size - 1);
    }
    
    void build(const vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
            return;
        }
        
        int mid = (start + end) / 2;
        build(arr, 2 * node, start, mid);
        build(arr, 2 * node + 1, mid + 1, end);
        tree[node] = tree[2 * node] + tree[2 * node + 1]; // Sum query
    }
    
    void propagate(int node, int start, int end) {
        if (lazy[node] != 0) {
            tree[node] += (end - start + 1) * lazy[node]; // Update node value
            
            if (start != end) {
                lazy[2 * node] += lazy[node];     // Mark child as lazy
                lazy[2 * node + 1] += lazy[node]; // Mark child as lazy
            }
            
            lazy[node] = 0; // Reset lazy value
        }
    }
    
    void updatePoint(int node, int start, int end, int idx, int val) {
        if (start == end) {
            tree[node] = val;
            return;
        }
        
        int mid = (start + end) / 2;
        if (idx <= mid)
            updatePoint(2 * node, start, mid, idx, val);
        else
            updatePoint(2 * node + 1, mid + 1, end, idx, val);
        
        tree[node] = tree[2 * node] + tree[2 * node + 1]; // Sum query
    }
    
    void updateRange(int node, int start, int end, int l, int r, int val) {
        propagate(node, start, end);
        
        if (start > end || start > r || end < l)
            return;
        
        if (start >= l && end <= r) {
            tree[node] += (end - start + 1) * val;
            
            if (start != end) {
                lazy[2 * node] += val;
                lazy[2 * node + 1] += val;
            }
            
            return;
        }
        
        int mid = (start + end) / 2;
        updateRange(2 * node, start, mid, l, r, val);
        updateRange(2 * node + 1, mid + 1, end, l, r, val);
        
        tree[node] = tree[2 * node] + tree[2 * node + 1];
    }
    
    int querySum(int node, int start, int end, int l, int r) {
        if (start > end || start > r || end < l)
            return 0;
        
        propagate(node, start, end);
        
        if (start >= l && end <= r)
            return tree[node];
        
        int mid = (start + end) / 2;
        int p1 = querySum(2 * node, start, mid, l, r);
        int p2 = querySum(2 * node + 1, mid + 1, end, l, r);
        
        return p1 + p2;
    }
    
    int queryMin(int node, int start, int end, int l, int r) {
        if (start > end || start > r || end < l)
            return INT_MAX;
        
        propagate(node, start, end);
        
        if (start >= l && end <= r)
            return tree[node];
        
        int mid = (start + end) / 2;
        int p1 = queryMin(2 * node, start, mid, l, r);
        int p2 = queryMin(2 * node + 1, mid + 1, end, l, r);
        
        return min(p1, p2);
    }
    
    int queryMax(int node, int start, int end, int l, int r) {
        if (start > end || start > r || end < l)
            return INT_MIN;
        
        propagate(node, start, end);
        
        if (start >= l && end <= r)
            return tree[node];
        
        int mid = (start + end) / 2;
        int p1 = queryMax(2 * node, start, mid, l, r);
        int p2 = queryMax(2 * node + 1, mid + 1, end, l, r);
        
        return max(p1, p2);
    }
};
```

**Quando usar**:
- Problemas que envolvem consultas em intervalos
- Problemas que envolvem atualizações em intervalos
- Quando você precisa encontrar o máximo, mínimo ou soma em um intervalo
- RSQ (Range Sum Query), RMQ (Range Minimum Query)

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

## Algoritmos prontos

1. **Codeforces 1140C - Playlist**

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>

using namespace std;

int main() {
    int n, k;
    cin >> n >> k;
    
    vector<pair<int, int>> musicas(n); 
    
    for (int i = 0; i < n; ++i) {
        int t, b;
        cin >> t >> b;
        musicas[i] = {b, t};
    }

    sort(musicas.rbegin(), musicas.rend());

    priority_queue<int, vector<int>, greater<int>> min_heap; 
    long long soma_duracoes = 0;
    long long max_prazer = 0;

    for (auto musica : musicas) {
        soma_duracoes += musica.second;
        min_heap.push(musica.second);

        if ((int)min_heap.size() > k) {
            soma_duracoes -= min_heap.top();
            min_heap.pop();
        }

        long long prazer = soma_duracoes * musica.first;
        max_prazer = max(max_prazer, prazer);
    }

    cout << max_prazer << endl;
    return 0;
}
```

2. **Codeforces 500B - New Year Permutation**

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
using namespace std;

void bfs(int start, vector<vector<int>>& adj, vector<bool>& visited, vector<int>& grupo) {
    queue<int> q;
    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        int atual = q.front();
        q.pop();
        grupo.push_back(atual);

        for (int vizinho : adj[atual]) {
            if (!visited[vizinho]) {
                visited[vizinho] = true;
                q.push(vizinho);
            }
        }
    }
}

int main() {
    int n;
    cin >> n;

    vector<int> p(n);
    for (int i = 0; i < n; ++i)
        cin >> p[i];

    vector<vector<int>> adj(n);
    for (int i = 0; i < n; ++i) {
        string linha;
        cin >> linha;
        for (int j = 0; j < n; ++j) {
            if (linha[j] == '1') {
                adj[i].push_back(j);
            }
        }
    }

    vector<bool> visited(n, false);
    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            vector<int> grupo;
            bfs(i, adj, visited, grupo);

            vector<int> valores;
            for (int idx : grupo)
                valores.push_back(p[idx]);

            sort(grupo.begin(), grupo.end());
            sort(valores.begin(), valores.end());

            for (int j = 0; j < grupo.size(); ++j)
                p[grupo[j]] = valores[j];
        }
    }

    for (int i = 0; i < n; ++i)
        cout << p[i] << " ";
    cout << endl;

    return 0;
}
```