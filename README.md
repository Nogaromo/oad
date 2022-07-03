# Домашнее задание № 7 (Бонусное)

## Машинное обучение

В этом задании нужно применить пакет [TensorFlow](https://www.tensorflow.org) для классификации событий (частиц) в эксперименте LHCb.

**Дедлайн 16 июня 23:55**

В блокноте Jupyter `particles.ipynb` находится пример кода, реализующий многоклассовый классификатор с использованием нейросетей.
Ваша задача модифицировать блокнот и подобрать параметры нейросети и обучения таким образом, чтобы показатель ROC AUC для случая «электрон против всего остального» был больше 0.97 на тестовой выборке.

## Система оценивания

Это домашнее задание бонусное. Это значит, что если его не сделать и не сдать, то накопленная оценка не изменится (не улучшится и не ухудшится).

За выполненное задание полагается 4 бонусных балла, плюс еще от 0 до 2 бонусных баллов пропорционально позиции в рейтинге, упорядоченной по значению ROC AUC вычисленному для лучая «электрон против всего остального».
