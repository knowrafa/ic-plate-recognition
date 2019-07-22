import os
from tabulate import tabulate

true_positive = 2000
false_negative = 300
false_positive = 340
true_negative = 300

results = []
results.append(("[Valor Real] Placas", true_positive, false_negative))
results.append(("[Valor Real] Não Placas",   false_positive, true_negative))
#print(tabulate(results, headers=[" ", "[Valor Predito] Placas", "[Valor Predito] Não Placas"]))
a = tabulate(results, headers=[" ", "[Valor Predito] Placas", "[Valor Predito] Não Placas"])
text_file = open("write_tabulate.txt", "w+")
text_file.write(a)
text_file.close()