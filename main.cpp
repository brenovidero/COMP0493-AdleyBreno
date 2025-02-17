#include <iostream>
#include "string_processor.h"

int main() {
    StringProcessor sp;
    
    // Exemplo de uso
    std::string text = sp.readUntilSevenDots("input.txt");
    std::cout << "Texto lido: " << text << "\n\n";
    
    // Encontrar ocorrências
    std::vector<int> positions = sp.findAllOccurrences(text, "love");
    std::cout << "Posições de 'love': ";
    for (int pos : positions) std::cout << pos << " ";
    std::cout << "\n\n";
    
    // Análise de caracteres
    auto analysis = sp.analyzeCharacters(text);
    std::cout << "Análise:\n";
    std::cout << "Dígitos: " << analysis.digits << "\n";
    std::cout << "Vogais: " << analysis.vowels << "\n";
    std::cout << "Consoantes: " << analysis.consonants << "\n\n";
    
    // Menor token
    std::cout << "Menor token: " << sp.findSmallestToken(text) << "\n\n";
    
    // Palavras mais frequentes
    auto frequent = sp.findMostFrequentWords(text);
    std::cout << "Palavras mais frequentes: ";
    for (const auto& word : frequent) std::cout << word << " ";
    std::cout << "\n\n";
    
    // Caracteres na última linha
    std::cout << "Caracteres na última linha: " << sp.countLastLineCharacters("input.txt") << "\n";
    
    return 0;
} 