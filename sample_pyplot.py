import matplotlib.pyplot as plt


X = [590,540,740,130,810,300,320,230,470,620,770,250]
Y = [.32,.36,.39,.52,.61,.72,.77,.75,.68,.57,.48,.48]

#scatter plot
#plt.clf()
#plt.clear()
plt.scatter(X, Y, s=60, c='red', marker='^')
#plt.clf()
#change axes ranges
plt.xlim(0,1000)
plt.ylim(0,1)

#add title
plt.title('Relationship Between Temperature and Iced Coffee Sales')

#add x and y labels
plt.xlabel('Cups of Iced Coffee Sold')
plt.ylabel('Temperature in Fahrenheit')

plt.subplot(2, 1, 2)

#show plot
plt.show()