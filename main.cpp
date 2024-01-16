#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <regex>
#include <fstream>

std::string exec(const char *cmd)
{
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe)
  {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
  {
    result += buffer.data();
  }
  return result;
}

std::vector<std::string> splitString(const std::string &str)
{
  std::istringstream iss(str);
  std::vector<std::string> tokens;
  std::string token;
  while (std::getline(iss, token, ' '))
  {
    if (!token.empty())
    {
      tokens.push_back(token);
    }
  }
  return tokens;
}

std::string readEmbeddings(const std::string filename)
{
  std::string fileContent;
  std::string line;
  std::ifstream embFile(filename);
  if (embFile.is_open())
  {
    while (getline(embFile, line))
    {
      fileContent += line;
    }
    embFile.close();
  }
  else
    std::cout << "Unable to open file " << filename;

  return fileContent;
}

std::vector<unsigned long long> parseEmbeddings(const std::vector<std::string> tokens)
{
  int vecSize = tokens.size(); // returns length of vector

  std::vector<unsigned long long> embeddings(vecSize);

  for (unsigned int i = 0; i < vecSize; i++)
  {
    try
    {
      embeddings.at(i) = std::stoull(tokens[i], nullptr, 0);
    }
    catch (...)
    {
      std::cout << "ERROR parsing " << tokens[i] << "\n";
    }
  }
  return embeddings;
}

int euclideanDistance(std::vector<unsigned long long> emb1, std::vector<unsigned long long> emb2)
{
  int distance = 0;
  for (unsigned int i = 0; i < emb1.size(); i++)
  {
    distance += (emb1.at(i) - emb2.at(i)) * (emb1.at(i) - emb2.at(i));
  }
  return distance;
}

int main()
{
  std::string cmd = "python main.py";

  std::cout << "Calling Python Model" << std::endl;

  std::string result = exec(cmd.c_str());

  std::cout << "Model Called" << std::endl;

  std::vector<std::string> tokens = splitString(result);

  std::vector<unsigned long long> live_embeddings = parseEmbeddings(tokens);

  // std::cout << result << std::endl;
  // std::cout << "!!!!!!!!!!!" << std::endl;

  int numOfFiles = 2;

  std::string files[2] = {"embedding/embedding_matteo.txt", "embedding/embedding_tommi.txt"};

  int minDistance = 0;
  int distances[2];

  for (int i = 0; i < 2; i++)
  {
    std::string file = files[i];
    std::string file_content = readEmbeddings(file);
    std::vector<std::string> file_tokens = splitString(file_content);
    std::vector<unsigned long long> file_embeddings = parseEmbeddings(file_tokens);

    // std::cout << file << std::endl;

    distances[i] = euclideanDistance(live_embeddings, file_embeddings);

    // std::cout << distances[i] << std::endl;
    // std::cout << "!!!!!!" << std::endl;
  }

  // auto it = std::min_element(std::begin(distances), std::end(distances));
  // std::cout << "index of smallest element: " << std::distance(std::begin(distances), it);
  int simIndex = std::distance(std::begin(distances), std::min_element(std::begin(distances), std::end(distances)));

  std::cout << "Face: " << files[simIndex] << std::endl;

  return 0;
}