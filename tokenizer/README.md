# NepaliBPE Tokenizer

- Tokenizer trained on (iriisnepal U nepberta) dataset
- [training code: ./1_train_tokenizer.ipynb](./1_train_tokenizer.ipynb)
- [data - preparation: ../dataset/0_download_data.ipynb](../dataset/0_download_data.ipynb)

## Example Usage

```
from transformers import PreTrainedTokenizerFast

# Example Nepali text
text = "गत आइतबार साँझपख केही आत्मीय मित्रहरूसँग भलाकुसारीमा थिएँ । गफैगफमा राजनीति र खेलकुदका प्रसंग उठे । त्यत्तिकैमा नेपालमा हुने गरेका साहित्य महोत्सव र वक्ताहरूका बारेमा एकछिन भुलियो । जयपुर साहित्य महोत्सवको कुरा पनि निस्कियो ।"

tokenizer = PreTrainedTokenizerFast.from_pretrained("Aananda-giri/NepaliBPE")
tokenizer.tokenize(text)

'''
# output:
['गत</w>', 'आइतबार</w>', 'साँझपख</w>', 'केही</w>', 'आत्मीय</w>', 'मित्र', 'हरूसँग</w>', 'भलाकु', 'सा', 'रीमा</w>', 'थिएँ</w>', '।</w>', 'ग', 'फै', 'ग', 'फमा</w>', 'राजनीति</w>', 'र</w>', 'खेलकुदका</w>', 'प्रसंग</w>', 'उठे</w>', '।</w>', 'त्यत्ति', 'कैमा</w>', 'नेपालमा</w>', 'हुने</w>', 'गरेका</w>', 'साहित्य</w>', 'महोत्सव</w>', 'र</w>', 'वक्ता', 'हरूका</w>', 'बारेमा</w>', 'एकछिन</w>', 'भु', 'लियो</w>', '।</w>', 'जय', 'पुर</w>', 'साहित्य</w>', 'महोत्सवको</w>', 'कुरा</w>', 'पनि</w>', 'निस्कियो</w>', '।</w>']
'''
```
