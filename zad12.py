import torch 
import numpy as np
import pandas as pd


#Задание 1
# 1. Создайте тензор x целочисленного типа, хранящий случайное значение
x = torch.randint(1, 10, (1,), dtype=torch.int32)
print(x)

# 2. Преобразуйте тензор к типу float32
x = x.to(dtype=torch.float32)
print(x)

# 3. Проведите с тензором x ряд операций:
x_pow = x ** 2
print(x_pow)

random_value = torch.rand(1) * 9 + 1  
x_ym = x_pow * random_value
print(x_ym)

x_ex = torch.exp(x_ym)
print(x_ex)

x_ex = torch.tensor([x_ex.item()], dtype=torch.float32, requires_grad=True)
x_ex.backward()
print(x_ex.grad)

#Задание 2
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# matplotlib для построения графиков
import matplotlib.pyplot as plt  
# Модуль для создания нейронных сетей
import torch.nn as nn  
# Модуль для оптимизации
import torch.optim as optim 
 
# Считываем данные
df = pd.read_csv('data.csv')

print('\nЗадание 2.')

y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
# возьмем три признака для обучения
 # теперь используем три признака
X = df.iloc[:, [0, 1, 2]].values 

# Преобразуем данные в тензоры PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)  # Признаки
# Целевые значения 
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  

# Модель
class SimpleNeuron(nn.Module):
    def __init__(self):
        super(SimpleNeuron, self).__init__()
        # Линейный слой с 3 входами и 1 выходом
        self.linear = nn.Linear(3, 1)  
    def forward(self, x):
        return self.linear(x)


model = SimpleNeuron()
# Среднеквадратичная ошибка
criterion = nn.MSELoss()  
# Стохастический градиентный спуск
optimizer = optim.SGD(model.parameters(), lr=0.01)  

weights_history = []
# Обучение модели
num_iterations = 100
for iteration in range(num_iterations):
    # Прямой проход
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    # Обратный проход и оптимизация
    optimizer.zero_grad()  
    loss.backward()  
    optimizer.step()  

    # Сохраняем веса на каждой 10-й итерации
    if (iteration + 1) % 10 == 0:
        weights_history.append([model.linear.weight.data.numpy().copy(), model.linear.bias.data.numpy().copy()])
    # Выводим данные каждые 10 итераций
    if (iteration + 1) % 10 == 0:
        print(f'Итерация [{iteration + 1}/{num_iterations}], Ошибка: {loss.item():.4f}')
        print(f'Веса: {model.linear.weight.data.numpy()}')
        print(f'Смещение: {model.linear.bias.data.numpy()}\n')

# После обучения визуализируем результаты
# Преобразуем предсказания в numpy для визуализации
with torch.no_grad():  
    predicted = model(X_tensor).numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], color='red', marker='o', label='Iris-setosa')
ax.scatter(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2], color='blue', marker='x', label='Iris-versicolor')
ax.set_xlabel('1 признак')
ax.set_ylabel('2 признак')
ax.set_zlabel('3 признак')

x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
z_min, z_max = X[:, 2].min(), X[:, 2].max()
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
ax.set_zlim([z_min, z_max])

plt.legend()

x1_range = np.linspace(x_min, x_max, 10)
x2_range = np.linspace(y_min, y_max, 10)
x1, x2 = np.meshgrid(x1_range, x2_range)

for i, (weight, bias) in enumerate(weights_history):
    x3 = -(weight[0, 0] * x1 + weight[0, 1] * x2 + bias[0]) / weight[0, 2]
    ax.plot_surface(x1, x2, x3, alpha=0.5, color='gray')
    plt.pause(1) 
    if i == len(weights_history) - 1:
        ax.text(x1_range[-1] - 0.3, x2_range[-1], x3[-1, -1], 'END', size=14, color='red')

plt.show()