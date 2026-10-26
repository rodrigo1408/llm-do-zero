# Origem dos dados

`municipios.txt` contém os nomes distintos dos municípios brasileiros, um por linha.

- **Fonte:** API de Localidades do IBGE — `https://servicodados.ibge.gov.br/api/v1/localidades/municipios?view=nivelado`
  (documentação: https://servicodados.ibge.gov.br/api/docs/localidades). Dados públicos.
- **Coleta:** `python scripts/fetch_ibge.py` (26/10/2026): 5.571 municípios, 5.298 nomes distintos
  (nomes repetidos em estados diferentes, como "bom jesus", aparecem uma vez só).
- **Normalização:** Unicode NFC, minúsculas, espaços simples. Acentos, hífens e apóstrofos são preservados
  (vocabulário de 41 caracteres + o token de fronteira `.`).

Não há dados pessoais neste corpus.
