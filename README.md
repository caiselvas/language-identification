# Language Identification using Character Trigrams

L'identificació del llenguatge és una tasca de processament del llenguatge huma (PLH, NLP en anglès). En aquest projecte, s'explora
l'ús de trigrames per tal de crear un model capaç de resoldre aquest problema. Addicionalment, s'han implementat i provat 3 funcions de suavitzat (_smoothing_) diferents: Lidstone, Linear Discounting i Absolute Discounting.

## Dataset
Les dades utilitzades com a corpus d'entrenament i de test provenen de *Wortschats Leipzig Corpora*, que conté texts en diferents llenguatges. Concretament s'han usat el castellà, l'italià, l'anglès, el francès, el neerlandès i l'alemany.

El train consisteix en 30.000 frases de cada llenguatge, mentre que el test 10.000.

## Evaluació
L'evaluació del model, en termes d'accuracy, és d'un 99.8932%, que es tradueix en tan sols 64 errors.
La matriu de confusió és la següent:

![image](https://github.com/caiselvass/Language_Identification/assets/117848447/127b53a6-21ef-46ea-b2e3-50516decb30e)

## Continguts
- **Original_langId**: Conté el dataset obtingut de *Wortschats Leipzig Corpora*.
- **Preprocessed_langId**: Conté els datasets preprocessats.
- **Weights**: Conté els paràmetres del model (tant pel test com pel validation).
- **Train.py**: Inclou codi pel preprocessat dels textos i per la creació dels arxius json.
- **Main.ipynb**: Notebook amb el validation i el test del model. Part principal.
- **Report**: Documentació detallada sobre les decisions preses, justificacions, resultats i conclusions.
- **Requeriments**: python 11.+ , sklearn, matplotlib, seaborn, nltk, (spacy en cas de detectar noms propis).

---
**Referències**
- *Wortschats Leipzig Corpora*: [Link to Dataset](https://example.com/leipzig_corpora)
- [Assignatura de PLH del GIA (UPC)](https://www.fib.upc.edu/ca/estudis/graus/grau-en-intelligencia-artificial/pla-destudis/assignatures/PLH-GIA)
