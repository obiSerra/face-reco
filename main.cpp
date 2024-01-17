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
#include <cmath>

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

std::vector<double> parseEmbeddings(const std::vector<std::string> tokens)
{
  int vecSize = tokens.size();

  std::vector<double> embeddings(vecSize);

  for (unsigned int i = 0; i < vecSize; i++)
  {
    try
    {
      embeddings.at(i) = std::stof(tokens[i]);
    }
    catch (...)
    {
      std::cout << "ERROR parsing " << tokens[i] << "\n";
    }
  }
  return embeddings;
}

int euclideanDistance(std::vector<double> emb1, std::vector<double> emb2)
{
  int distance = 0;
  for (unsigned int i = 0; i < emb1.size(); i++)
  {
    distance += (emb1.at(i) - emb2.at(i)) * (emb1.at(i) - emb2.at(i));
  }
  return distance;
}

double cosine_distance(
    std::vector<double> &vec_a,
    std::vector<double> &vec_b)
{
  int vec_size = vec_a.size();
  double a_dot_b = 0.0;
  double a_mag = 0;
  double b_mag = 0;

  for (size_t i = 0; i < vec_size; ++i)
  {
    // std::cout << vec_a[i] << " " << vec_b[i] << std::endl;
    a_dot_b += (vec_a[i] * vec_b[i]);
    a_mag += (vec_a[i] * vec_a[i]);
    b_mag += (vec_b[i] * vec_b[i]);
  }
  double dist = 1.0 - (a_dot_b / (sqrt(a_mag) * sqrt(b_mag)));

  return dist;
}

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
    return 1;
  }

  int numOfFiles = 2;

  // std::string files[2] = {"embedding/embedding_matteo.txt", "embedding/embedding_tommi.txt", "embedding/embedding_robi.txt"};
  std::string files[2] = {"embedding/embedding_matteo.txt", "embedding/embedding_tommi.txt"};

  std::string image_path = argv[1];

  std::string cmd = "python main.py " + image_path;

  std::cout << "Calling Python Model" << std::endl;

  std::string result = exec(cmd.c_str());

  std::cout << "Model Called" << std::endl;

  std::vector<std::string> tokens = splitString(result);

  std::vector<double> live_embeddings = parseEmbeddings(tokens);

  // std::cout << "!!!!!!!!!!!" << std::endl;

  int minDistance = 0;
  double distances[2];

  for (int i = 0; i < 2; i++)
  {
    std::string file = files[i];
    std::string file_content = readEmbeddings(file);
    std::vector<std::string> file_tokens = splitString(file_content);
    std::vector<double> file_embeddings = parseEmbeddings(file_tokens);

    distances[i] = cosine_distance(live_embeddings, file_embeddings);
  }

  int simIndex = std::distance(std::begin(distances), std::min_element(std::begin(distances), std::end(distances)));

  std::cout << "Face: " << files[simIndex] << std::endl;

  return 0;
}