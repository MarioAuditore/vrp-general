from dataclasses import dataclass, field
from typing import Generic, List, Tuple, TypeVar

"""
Описание основных моделей данных для модели
"""

__all__ = [
    'Node',
    'Cargo',
    'Tariff',
    'TariffCost',
    'Route'
]


@dataclass(
    frozen=True,
    slots=True
)
class Node:
    """
        Описание точки посещения
    """
    id: object
    cargo_id: object
    """
        Количество человек (положительна, если загрузка, отрицательна для разгрузки)
    """
    capacity: int = field(hash=True)
    """
        Время начало поездки
    """
    start_time: int = field(hash=False)
    end_time: int = field(hash=False)
    service_time: int = field(hash=True)
    """
        Координаты для адреса
    """
    coordinates: Tuple[float, float] = field(hash=False)


@dataclass(
    frozen=True,
    slots=True
)
class Cargo:
    """
    Класс, который описывает груз
    """
    """
        Идентификатор груза
    """
    id: object = field(hash=True)
    """
        Список точек для посещения (начальная и конечная)
    """
    nodes: List[Node] = field(hash=False)


@dataclass
class TariffCost:
    # минимальное расстояние по текущему тарифу
    min_dst_km: float = field()
    # максимальное расстояние по текущему тарифу
    max_dst_km: float = field()
    # цена за километр в рублях
    cost_per_km: float = field()
    # фиксированная цена в рублях
    fixed_cost: float = field()


@dataclass(
    frozen=True,
    slots=True
)
class Tariff:
    # идентификатор тарифа (например "эконом")
    id: object = field()
    # вместимость машин на данном тарифе
    capacity: int = field(hash=False)

    """
        список "сегментов" - цен на разном километраже.
        Например, если тариф до 5км фиксированный (100р.), а после 5 км тариф линейный (100 * км + 500), то в Tariff
        задается так:
        Tariff(
                id='эконом',
                capacity=4,
                cost_per_distance=[
                    TariffCost(min_dst_km=0, max_dst_km=5, cost_per_km=0, fixed_cost=100),
                    TariffCost(min_dst_km=5, max_dst_km=float('inf'), cost_per_km=100, fixed_cost=500)
                ]
            )
    """
    cost_per_distance: List[TariffCost] = field(hash=False)

    """
        Максимальное кол-во ТС данного типа, -1 для неограниченного использования.
        Если тариф это тариф для личного транспорта, то поле игнорируется и используется ровно столько, сколько 
        передано заявок на личный транспорт

    """
    max_count: int = field(hash=False, default=-1)


T = TypeVar('T')  # хз зачем тут это, но по приколу оставлю


@dataclass
class Route(Generic[T]):
    """
        Класс маршрута
    """
    id: int = field(hash=True)  # идентификатор маршрута
    path: list[T] = field(hash=False)  # путь в нодах или любой другом типе
    tariff: Tariff = field(hash=False)  # тип машины на данном маршруте
