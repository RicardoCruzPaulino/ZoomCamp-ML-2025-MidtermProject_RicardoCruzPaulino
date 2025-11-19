import requests 

url = 'http://127.0.0.1:9696/predict'

customer = {
  "workclass": "Private",
  "education": "HS-grad",
  "marital-status": "Married-civ-spouse",
  "occupation": "Prof-specialty",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "native-country": "United-States",
  #"income": "<=50K",
  "age": 17,
  "fnlwgt": 12285,
  "education_num": 1,
  "capital-gain": 99999,
  "capital-loss": 4356,
  "hours-per-week": 1
}

response = requests.post(url, json=customer)

if response.status_code == 200:
    try:
        income = response.json()
    except ValueError:
        print("La respuesta no es JSON válida.")
else:
    print(f"Error HTTP {response.status_code}: {response.text}")


print('response:', income)

if  income >= 0.5:
    print('send email with Credit card promo')
else:
    print('Don\'t do anything')
 