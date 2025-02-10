# %%
# Устанавливаем необходимые бибилиотеки в систему.
#!pip install aiofiles
#!pip install prophet
#!pip install pandas
#!pip install matplotlib
#!pip install numpy
#!pip install aiogram

# %%
# Подгружаем необходимые библиотеки
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from matplotlib.dates import DateFormatter
import asyncio
from matplotlib import rcParams
from prophet import Prophet
from data.TOKEN import API_TOKEN, API_KEY
import aiofiles
import aiohttp

# Настройка шрифта для графиков
rcParams["font.family"] = "DejaVu Sans"
plt.style.use("seaborn-v0_8")

# Создание бота и диспетчера событий
bot = Bot(token=API_TOKEN)
dp = Dispatcher()


# %%
# API_KEY ключ для получения данных с сайта mos.ru
URL = "https://apidata.mos.ru/v1/datasets/2453/rows"
TOP = 1000 # количество строк в запросе к сайту. Такое ограничение указывает сайт mos.ru
BATCH_SIZE = 10 # количество одновременных подключений к сайту

# Инициализация пустого списка для хранения всех данных
all_data = []

# Создание асинхронной сессии для выполнения HTTP-запросов
async with aiohttp.ClientSession() as session:
    # Инициализация переменной для отслеживания текущей страницы
    page_index = 0

    # Бесконечный цикл для получения данных, пока они доступны
    while True:
        # Создание списка задач для параллельного выполнения запросов
        tasks = []
        for i in range(BATCH_SIZE):
            # Вычисление значения параметра $skip для пагинации
            skip = (page_index + i) * TOP
            # Добавление задачи (запроса) в список задач
            tasks.append(fetch_page(session, skip))

        # Вывод информации о текущем диапазоне запрашиваемых страниц
        print(
            f"Запрашиваем страницы с {(page_index)*TOP} до {((page_index+BATCH_SIZE-1)*TOP)}"
        )

        # Параллельное выполнение всех задач и ожидание их завершения
        results = await asyncio.gather(*tasks)

        # Флаг для остановки цикла, если данные закончились
        stop = False
        for res in results:
            # Если результат пустой (например, достигнут конец данных)
            if not res:
                stop = True
                break  # Выход из цикла for

            # Добавление полученных данных в общий список
            all_data.extend(res)

        # Если достигнут конец данных, выход из основного цикла
        if stop:
            print("Достигнут конец данных.")
            break  # Выход из цикла while

        # Увеличение индекса страницы для следующей пачки запросов
        page_index += BATCH_SIZE

    # Преобразование списка всех данных в DataFrame
    df = pd.DataFrame(all_data)

    # Если в DataFrame есть столбец "Cells", нормализуем его (распаковываем JSON)
    if "Cells" in df.columns:
        df = pd.json_normalize(df["Cells"])

# %%

# очистка данных
cp_data=df.copy(deep=True)
cp_data = cp_data.drop(0, axis=0) #удаляем первую строку, с названиями колонок, но при переносе по АПИ их несоздается
cp_data = cp_data.drop(['ID', 'global_id', 'StationName', 'AdmArea', 'SurveillanceZoneCharacteristics', 'Location', 'MonthlyAveragePDKss'], axis=1)
cp_data['Parameter'].value_counts()

# %%
cp_data = cp_data[cp_data['Parameter'].isin(['Диоксид азота', 'Озон', 'Взвешенные частицы РМ10', 'Диоксид серы'])]
cp_data = cp_data.reset_index()
del cp_data['index']

# %%
cp_data['PDKnorm'] = np.nan
cp_data['PDKnorm'] = cp_data['PDKnorm'].mask(cp_data['Parameter'] == 'Диоксид азота', 0.04)
cp_data['PDKnorm'] = cp_data['PDKnorm'].mask(cp_data['Parameter'] == 'Озон', 0.03)
cp_data['PDKnorm'] = cp_data['PDKnorm'].mask(cp_data['Parameter'] == 'Взвешенные частицы РМ10', 0.04)
cp_data['PDKnorm'] = cp_data['PDKnorm'].mask(cp_data['Parameter'] == 'Диоксид серы', 0.05)

# %%
cp_data['MonthlyAverage'] = cp_data['MonthlyAverage'].astype('Float64')
cp_data['Period'] = pd.to_datetime(cp_data['Period'], format='%m.%Y', errors='coerce')
#После обработки сохраняем данный в файл
cp_data.to_csv('new_data.csv')

# %%
#очищенные данные копируев в изначальный ДатаФрэйм для дальнейшей обработки и вывода графиков
df=cp_data.copy(deep=True)

# %%
# Чтение данных из CSV-файла когда создавали вручную, либо если будет условие, проверки даты создания файла, на устаревания
#csv_file = "new_data.csv"
#df = pd.read_csv(csv_file)
#df["Period"] = pd.to_datetime(df["Period"], errors="coerce")  # Преобразование даты

# %%
# Удаление строк с отсутствующими критическими значениями
df = df.dropna(subset=["District", "Period", "MonthlyAverage", "PDKnorm", "Parameter"])

# Предобработка данных для прогнозов
preprocessed_data = {
    district: {
        param: group[["Period", "MonthlyAverage"]]
        .rename(columns={"Period": "ds", "MonthlyAverage": "y"})
        .sort_values("ds")
        for param, group in df[df["District"] == district].groupby("Parameter")
    }
    for district in df["District"].unique()
}

# Получение уникальных районов для клавиатуры
districts = df["District"].unique().tolist()
keyboard = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text=d)] for d in districts], resize_keyboard=True
)


# %%
# Кэш моделей для прогнозов
model_cache = {}
cache_ttl = 3600  # Время жизни кэша в секундах

# %%
# Функция для удаления выбросов из данных
def remove_outliers(df, sigma=3):
    """Удаление выбросов с использованием стандартного отклонения"""
    mean = df["y"].mean()
    std_dev = df["y"].std()
    return df[(df["y"] >= mean - sigma * std_dev) & (df["y"] <= mean + sigma * std_dev)]


# Асинхронная функция для создания прогноза и управления кэшем
async def create_forecast(district, param):
    """Асинхронное создание прогноза с кэшированием и управлением временем жизни кэша"""
    cache_key = f"{district}_{param}"

    # Проверка наличия прогноза в кэше и его актуальности
    if cache_key in model_cache:
        model, forecast, timestamp = model_cache[cache_key]
        if asyncio.get_event_loop().time() - timestamp < cache_ttl:
            return model, forecast

    data = preprocessed_data.get(district, {}).get(param)
    if data is None or len(data) < 24:
        retujupyter nbconvert --to script ваш_ноутбук.ipynbrn None, None

    try:
        clean_data = remove_outliers(data)

        # Создание и обучение модели Prophet
        model = Prophet(
            changepoint_prior_scale=0.15,
            seasonality_prior_scale=15,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.80,
        )
        model.fit(clean_data)

        # Построение будущих значений для прогноза
        future = model.make_future_dataframe(periods=24, freq="ME")
        forecast = model.predict(future)

        # Сохранение прогноза в кэш
        model_cache[cache_key] = (model, forecast, asyncio.get_event_loop().time())
        return model, forecast

    except Exception:
        return None, None


# %%
# Обработчик команды /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Выберите район:", reply_markup=keyboard)


# %%
# Обработчик выбора района и построения графиков
@dp.message(F.text.in_(districts))
async def send_combined_plot(message: types.Message):
    district = message.text

    if district not in preprocessed_data:
        return await message.answer("Данные по району не найдены")

    parameters = list(preprocessed_data[district].keys())
    if not parameters:
        return await message.answer("Нет данных по параметрам загрязнения")

    num_plots = len(parameters)
    fig, axs = plt.subplots(
        num_plots, 1, figsize=(12, 5 * num_plots) if num_plots > 1 else (12, 5)
    )
    plt.subplots_adjust(hspace=0.8)

    if num_plots == 1:
        axs = [axs]

    colors = plt.cm.tab10.colors
    caption = f"Анализ и прогноз для {district}"

    tasks = []
    for param in parameters:
        tasks.append(create_forecast(district, param))

    results = await asyncio.gather(*tasks)

    for i, ((model, forecast), param) in enumerate(zip(results, parameters)):
        ax = axs[i]
        color = colors[i % len(colors)]
        data = preprocessed_data[district][param]

        # Построение исторических данных
        ax.plot(
            data["ds"],
            data["y"],
            color=color,
            linewidth=1.5,
            label="Исторические данные",
        )

        if forecast is not None:
            forecast_period = forecast[forecast["ds"] > data["ds"].max()]
            ax.plot(
                forecast_period["ds"],
                forecast_period["yhat"],
                "purple",
                linestyle="--",
                label="Прогноз",
            )

        # Построение уровня ПДК
        pdk_value = df[(df["District"] == district) & (df["Parameter"] == param)][
            "PDKnorm"
        ].iloc[0]
        ax.axhline(pdk_value, color="red", linestyle="-.", label="ПДК")
        ax.set_title(f"{district} - {param}", fontsize=12, pad=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(DateFormatter("%Y.%m"))
        # Наклон подписей дат на 45 градусов
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Сохранение графика и отправка пользователю
    filename = f"{district}_forecast.png"
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close()

    # Асинхронная работа с файлами
    async with aiofiles.open(filename, mode="rb") as file:
        await message.answer_photo(
            types.BufferedInputFile(await file.read(), filename=filename),
            caption=caption,
        )

    os.remove(filename)  # Удаление временного файла


# %%
# Главная функция для запуска бота
async def main():
    await dp.start_polling(bot)


# %%
# Точка входа в приложение
if __name__ == "__main__":
    await main()



