# Exata-Falada

## Problemas Conhecidos:
* Arquivos grandes ficam lagados (tentar não renderizar tudo de uma vez, especialmente o MathJax)
* PDF com senha não funciona
* Leitor nativo do Edge não lê a descrição das equações
* Algumas equações não estão sendo centralizadas. Não estão usando a tag correta (\$\$ ... \$\$)

## Chaves do .env:
* GOOGLE_API_KEY
##### Opcionais (Modo BUCKET):
* Bucket AWS S3:
  * AWS_ACCESS_KEY_ID
  * AWS_SECRET_ACCESS_KEY
  * S3_BUCKET
  * S3_REGION

* Bucket Oracle Object Storage:
  * OCI_ACCESS_KEY_ID
  * OCI_SECRET_ACCESS_KEY
  * OCI_BUCKET
  * OCI_REGION
  * OCI_NAMESPACE
  * OCI_ENDPOINT_URL
