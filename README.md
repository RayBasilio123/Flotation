# Flotation

## Arquivos de Dados

*   **[`dados/Flotacao_Dados_Final.csv`](dados/Flotacao_Dados_Final.csv)**: Contém os dados originais, sem alterações significativas aplicadas após a concatenação inicial.
*   **[`dados/flotacao_dados_dentro_do_padrao.csv`](dados/flotacao_dados_dentro_do_padrao.csv)**: Contém os dados filtrados, mantendo apenas os registros onde houve uma mudança no valor de `conc_silica` exatamente a cada duas horas e excluindo períodos onde a usina não estava em operação (filtro `operacao == True`).
*   **[`codigo/data_clean.ipynb`](codigo/data_clean.ipynb)**: Notebook contendo o código para a limpeza dos dados e a concatenação dos arquivos fornecidos.
*   **[`codigo/colab/vale10_04_2025.ipynb`](codigo/colab/vale10_04_2025.ipynb)**: Notebook com a previsão de sílica.
