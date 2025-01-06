# AI-Lab

Програма, що дає користувачу можливість побудувати свою модель нейронної мережі прямо у програмі і там же її натренувати(ПРОГРАМА НЕ ДОРОБЛЕНА).

Щоб запустити код програми, активуйте env (conda activate /opt/anaconda3/envs/tf_test_cpu) У ВАС ПОВИННА БУТИ ВСТАНОВЛЕНЯ ANACONDA.

Далі запустіть python main.py

У папці assets/mnist ви можете знайти .png малюнки, за допомогою яких можна перевірити, наскільки правильно була натренована нейромережа на датасеті mnist(у самій програмі впевніться, що при виборі файлу ви поставили розширення .png).

У папці assets/cifar10 ви можете знайти .png малюнки, за допомогою яких можна перевірити, наскільки правильно була натренована нейромережа на датасеті cifar10

У папці datasets ви можете знайти train та test датасет, на якому нейромережа може навчитись розпізнавати, на малюнку є кіт чи ні (малюнки здається 64 на 64).

От приклад моделі нейронної мережі, для датасету mnist:

![Image alt](https://github.com/yvsazh/AI-Lab/raw/main/forGitHub/mnist_model.jpg)

Таким чином виглядає тренування:

Використаємо малюнок:
![Image alt](https://github.com/yvsazh/AI-Lab/raw/main/assets/mnist/82.png)

Отримаємо відповідь нейромережі:
![Image alt](https://github.com/yvsazh/AI-Lab/raw/main/forGitHub/prediction82.jpg)

Отаким чином налаштовуємо свій датасет .h5:

![Image alt](https://github.com/yvsazh/AI-Lab/raw/main/forGitHub/myDatasetExample.jpg)


