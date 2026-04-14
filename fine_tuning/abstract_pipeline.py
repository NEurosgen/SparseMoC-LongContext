from abc import ABC, abstractmethod
import torch
import logging
import time
import json
import os
from datetime import datetime


class Pipeline(ABC):
    """
    Базовый класс пайплайна с продвинутым логированием.
    
    Каждый подкласс получает:
    - Python logger
    - JSON-метрики
    - Тайминг каждого этапа
    """

    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._metrics_history = []
        self._start_time = None

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            fmt = logging.Formatter(
                '[%(asctime)s] %(name)s | %(levelname)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            ch.setFormatter(fmt)
            self.logger.addHandler(ch)

            log_dir = getattr(self.config, 'log_dir', None)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                fh = logging.FileHandler(
                    os.path.join(log_dir, f'{self.__class__.__name__}.log'),
                    encoding='utf-8'
                )
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(fmt)
                self.logger.addHandler(fh)

    def run_process(self, model, data, profiler=None):
        """Запуск этапа с автоматическим логированием времени."""
        name = self.__class__.__name__
        self.logger.info(f'{"="*60}')
        self.logger.info(f'Starting: {name}')
        self.logger.info(f'{"="*60}')
        self._start_time = time.time()

        result = self.process(model, data, profiler)

        elapsed = time.time() - self._start_time
        self.logger.info(f'Completed: {name} in {elapsed:.1f}s ({elapsed/60:.1f}min)')
        self.log_result(result)
        self._save_metrics_json()
        self.teardown()
        return result

    def log_metric(self, **kwargs):
        """
        Залогировать метрики. Автоматически добавляет timestamp.
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_s': round(time.time() - self._start_time, 2) if self._start_time else 0,
            **kwargs
        }
        self._metrics_history.append(entry)

    def _save_metrics_json(self):
        """Сохранить все метрики в JSON-файл."""
        log_dir = getattr(self.config, 'log_dir', './logs')
        os.makedirs(log_dir, exist_ok=True)
        
        metrics_file = os.path.join(
            log_dir, f'{self.__class__.__name__}_metrics.json'
        )
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self._metrics_history, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f'Metrics saved to {metrics_file} ({len(self._metrics_history)} entries)')

    @abstractmethod
    def process(self, model, data, profiler=None):
        pass

    @abstractmethod
    def log_result(self, result):
        pass

    def teardown(self):
        """Очистка ресурсов после завершения."""
        pass
