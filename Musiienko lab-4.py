import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score


print('1. Відкрити та зчитати файл з даними.')
data = pd.read_csv('dataset2_l4.txt')

print(f'2. Визначити та вивести кількість записів: {data.shape[0]}')
print(f'3. Вивести атрибути набору даних: {",".join(data.columns)}')

print('4. З’ясувати збалансованість набору даних.')
class_counts = data['Class'].value_counts()
print(class_counts) # Кількість об'єктів кожного класу відрізняється, ми можемо вважати набір даних незбалансованим.


print('5. Отримати двадцять варіантів перемішування набору даних та розділення його на навчальну (тренувальну) та тестову вибірки,', 
      'використовуючи функцію ShuffleSplit. Сформувати начальну та тестові вибірки на основі обраного користувачем варіанту.')

shuffle_split = ShuffleSplit(n_splits=20, test_size=0.2)

split_variants = []
for train_index, test_index in shuffle_split.split(data):
    train_df = data.iloc[train_index]
    test_df = data.iloc[test_index]
    split_variants.append((train_df, test_df))

variant = 5
train_df_selected, test_df_selected = split_variants[variant]
print("TRAIN:\n", train_df_selected.head(10))
print("TEST:\n", test_df_selected.head(10))


print('6. Використовуючи функцію KNeighborsClassifier бібліотеки scikit-learn, збудувати класифікаційну модель на основі', 
      'методу k найближчих сусідів (кількість сусідів обрати самостійно, вибір аргументувати) та навчити її на тренувальній вибірці,', 
      'вважаючи, що цільова характеристика визначається стовпчиком Class, а всі інші виступають в ролі вихідних аргументів.')

k = 3
k_neighbors_classifier = KNeighborsClassifier(n_neighbors=k)

x_train = train_df_selected.drop(columns=['Class'])
y_train = train_df_selected['Class']
k_neighbors_classifier.fit(x_train, y_train)

x_test = test_df_selected.drop(columns=['Class'])
y_test = test_df_selected['Class']


print('7. Обчислити класифікаційні метрики збудованої моделі для тренувальної та тестової вибірки.', 
      'Представити результати роботи моделі на тестовій вибірці графічно.')

def calculate_metrics(model, x_cord, y_cord):
    all_metrics = {'accuracy': 0, 
                   'precision': 0, 
                   'recall': 0, 
                   'f_scores': 0,
                   'MCC': 0, 
                   'BA': 0}

    model_predictions = model.predict(x_cord)
    all_metrics['accuracy'] = accuracy_score(y_cord, model_predictions)
    all_metrics['precision'] = precision_score(y_cord, model_predictions, average='weighted')
    all_metrics['recall'] = recall_score(y_cord, model_predictions, average='weighted')
    all_metrics['f_scores'] = f1_score(y_cord, model_predictions, average='weighted')
    all_metrics['MCC'] = matthews_corrcoef(y_cord, model_predictions)
    all_metrics['BA'] = balanced_accuracy_score(y_cord, model_predictions)

    return all_metrics


metrics_test_df = calculate_metrics(k_neighbors_classifier, x_test, y_test)
metrics_train_df = calculate_metrics(k_neighbors_classifier, x_train, y_train)

df_test_train_graph = pd.DataFrame({'Test': metrics_test_df, 'Train': metrics_train_df})
df_test_train_graph.plot(kind='bar', figsize=(10, 6))
plt.title('Метрики для тестової та тренувальної вибірок')
plt.ylabel('Значення')
plt.xlabel('Метрика')
plt.xticks(rotation=45)
plt.savefig(f'metrics.png')


print('8. Обрати алгоритм KDTree та з’ясувати вплив розміру листа (від 20 до 200 з кроком 5)', 
      'на результати класифікації. Результати представити графічно.')

f_scores_values = []
mcc_scores_values = []
ba_scores_values = []
leaf_sizes = range(20, 201, 5)

for leaf_size in leaf_sizes:
    model = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree', leaf_size=leaf_size)
    model.fit(x_train, y_train)

    model_predictions = model.predict(x_test)
    f_scores = f1_score(y_test, model_predictions, average='weighted')
    mcc_scores = matthews_corrcoef(y_test, model_predictions)
    ba_scores = balanced_accuracy_score(y_test, model_predictions)

    f_scores_values.append(f_scores)
    mcc_scores_values.append(mcc_scores)
    ba_scores_values.append(ba_scores)


plt.figure(figsize=(10, 6))
plt.plot(leaf_sizes, f_scores_values, linestyle='-')
plt.plot(leaf_sizes, mcc_scores_values, linestyle='-')
plt.plot(leaf_sizes, ba_scores_values, linestyle='-')
plt.xlabel('Розмір листа')
plt.ylabel('Metric Value')
plt.xticks(np.arange(min(leaf_sizes), max(leaf_sizes)+1, 10), rotation=45)
plt.grid()
plt.savefig(f'Metrics vs Розмір листа.png')
