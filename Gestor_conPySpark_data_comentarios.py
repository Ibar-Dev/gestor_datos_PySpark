"""

Un sistema de gestión de datos de nivel profesional construido sobre PySpark, implementando mejores prácticas
para el procesamiento de datos a gran escala, manejo de errores y optimización del rendimiento.

Características Clave:
- Operaciones seguras con manejo de errores integral
- Validación avanzada de datos y aplicación de esquemas
- Operaciones optimizadas en memoria para grandes conjuntos de datos
- Sistema de registro de logs completo
- Soporte para múltiples formatos de archivo con validación de esquemas
- Funciones de monitoreo y optimización del rendimiento
"""
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass
from functools import wraps
from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.types import StructType, DataType


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class MetricasDataFrame:
    """Almacena métricas de análisis de DataFrame para monitoreo de rendimiento."""
    cantidad_filas: int
    cantidad_columnas: int
    uso_memoria: str
    esquema: Dict[str, str]
    conteo_nulos: Dict[str, int]
    tiempo_ejecucion: float

class GestorDatosSpark:
    """
    Sistema avanzado de gestión de datos para procesamiento de datos a gran escala usando PySpark.
    
    Características:
    - Validación y aplicación automática de esquemas
    - Operaciones de datos optimizadas para el rendimiento
    - Manejo de errores y logging integral
    - Monitoreo del uso de memoria
    - Soporte para múltiples formatos de archivo con validación personalizada
    """
    
    FORMATOS_SOPORTADOS = {
        'csv': {'metodo_lectura': 'csv', 'metodo_escritura': 'csv'},
        'parquet': {'metodo_lectura': 'parquet', 'metodo_escritura': 'parquet'},
        'json': {'metodo_lectura': 'json', 'metodo_escritura': 'json'},
        'excel': {
            'metodo_lectura': 'com.crealytics.spark.excel',
            'metodo_escritura': 'com.crealytics.spark.excel'
        }
    }

    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        nombre_app: str = "GestorDatosSpark",
        nivel_log: int = logging.INFO
    ):
        """
        Inicializa GestorDatosSpark con configuración personalizada.
        
        Args:
            spark: SparkSession existente o None para crear una nueva
            nombre_app: Nombre para la aplicación Spark
            nivel_log: Nivel de logging (por defecto: INFO)
        """
        self.logger = logging.getLogger(nombre_app)
        self.logger.setLevel(nivel_log)
        
        self.spark = spark or self._crear_sesion_optimizada(nombre_app)
        self.datos: Optional[DataFrame] = None
        self.esquema: Optional[StructType] = None
        self._metricas_rendimiento: List[MetricasDataFrame] = []

    def _crear_sesion_optimizada(self, nombre_app: str) -> SparkSession:
        """Crea una SparkSession optimizada con configuraciones recomendadas."""
        return (SparkSession.builder
                .appName(nombre_app)
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                .config("spark.sql.shuffle.partitions", "auto")
                .config("spark.memory.offHeap.enabled", "true")
                .config("spark.memory.offHeap.size", "2g")
                .getOrCreate())

    def monitor_rendimiento(func: Callable) -> Callable:
        """Decorador para monitorear el rendimiento de las operaciones de DataFrame."""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            tiempo_inicio = datetime.now()
            resultado = func(self, *args, **kwargs)
            tiempo_ejecucion = (datetime.now() - tiempo_inicio).total_seconds()
            
            if isinstance(resultado, DataFrame):
                metricas = self._calcular_metricas(resultado, tiempo_ejecucion)
                self._metricas_rendimiento.append(metricas)
                self.logger.info(
                    f"La operación '{func.__name__}' se completó en {tiempo_ejecucion:.2f}s. "
                    f"Uso de memoria: {metricas.uso_memoria}"
                )
            return resultado
        return wrapper

    def _calcular_metricas(self, df: DataFrame, tiempo_ejecucion: float) -> MetricasDataFrame:
        """Calcula métricas completas para las operaciones de DataFrame."""
        return MetricasDataFrame(
            cantidad_filas=df.count(),
            cantidad_columnas=len(df.columns),
            uso_memoria=self._estimar_uso_memoria(df),
            esquema={field.name: str(field.dataType) for field in df.schema.fields},
            conteo_nulos=self._contar_nulos(df),
            tiempo_ejecucion=tiempo_ejecucion
        )

    def _estimar_uso_memoria(self, df: DataFrame) -> str:
        """Estima el uso de memoria del DataFrame en formato legible."""
        bytes_por_fila = sum(
            len(field.name) + self._obtener_tamano_tipo(field.dataType)
            for field in df.schema.fields
        )
        total_bytes = bytes_por_fila * df.count()
        return self._formatear_bytes(total_bytes)

    @staticmethod
    def _obtener_tamano_tipo(dtype: DataType) -> int:
        """Estima el tamaño en bytes para diferentes tipos de datos."""
        tamano_tipos = {
            'string': 40,  # Tamaño promedio de cadena
            'integer': 4,
            'long': 8,
            'double': 8,
            'boolean': 1,
            'date': 4,
            'timestamp': 8
        }
        return tamano_tipos.get(dtype.simpleString().lower(), 8)

    @staticmethod
    def _formatear_bytes(bytes_: int) -> str:
        """Convierte bytes a formato legible."""
        for unidad in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_ < 1024:
                return f"{bytes_:.2f}{unidad}"
            bytes_ /= 1024
        return f"{bytes_:.2f}PB"

    @monitor_rendimiento
    def leer_archivo(
        self,
        ruta_archivo: str,
        esquema: Optional[StructType] = None,
        validar_esquema: bool = True,
        **opciones
    ) -> Optional[DataFrame]:
        """
        Lee datos de un archivo con manejo avanzado de errores y validación de esquemas.
        
        Args:
            ruta_archivo: Ruta al archivo de entrada
            esquema: Esquema opcional para aplicar
            validar_esquema: Si se debe validar contra el esquema proporcionado
            **opciones: Opciones adicionales de lectura
            
        Returns:
            DataFrame o None si la operación falla
        """
        try:
            formato_archivo = self._detectar_formato_archivo(ruta_archivo)
            if formato_archivo not in self.FORMATOS_SOPORTADOS:
                raise ValueError(f"Formato de archivo no soportado: {formato_archivo}")

            lector = self.spark.read.format(
                self.FORMATOS_SOPORTADOS[formato_archivo]['metodo_lectura']
            )
            
            # Aplicar opciones predeterminadas para el formato
            lector = self._aplicar_opciones_formato(lector, formato_archivo)
            
            # Aplicar opciones personalizadas
            for clave, valor in opciones.items():
                lector = lector.option(clave, valor)

            if esquema:
                lector = lector.schema(esquema)

            self.datos = lector.load(ruta_archivo)
            self.esquema = self.datos.schema

            if validar_esquema and esquema:
                self._validar_esquema(esquema)

            self.logger.info(
                f"Datos cargados exitosamente desde {ruta_archivo}. "
                f"Filas: {self.datos.count()}, Columnas: {len(self.datos.columns)}"
            )
            return self.datos

        except Exception as e:
            self.logger.error(f"Error al leer el archivo {ruta_archivo}: {str(e)}")
            raise

    def _detectar_formato_archivo(self, ruta_archivo: str) -> str:
        """Detecta el formato del archivo a partir de la extensión con validación."""
        extension = os.path.splitext(ruta_archivo)[1].lower()[1:]
        if not extension:
            raise ValueError("El archivo no tiene extensión")
        return extension

    def _aplicar_opciones_formato(self, lector, formato_archivo: str):
        """Aplica opciones predeterminadas específicas del formato."""
        if formato_archivo == 'csv':
            return lector.option("header", "true") \
                       .option("inferSchema", "true") \
                       .option("mode", "PERMISSIVE")
        elif formato_archivo == 'excel':
            return lector.option("header", "true") \
                       .option("inferSchema", "true")
        return lector

    def _validar_esquema(self, esquema_esperado: StructType):
        """Valida el esquema del DataFrame contra el esquema esperado."""
        campos_actuales = set(f.name for f in self.datos.schema.fields)
        campos_esperados = set(f.name for f in esquema_esperado.fields)
        
        if campos_actuales != campos_esperados:
            faltantes = campos_esperados - campos_actuales
            adicionales = campos_actuales - campos_esperados
            raise ValueError(
                f"Desajuste de esquema. Campos faltantes: {faltantes}. Campos adicionales: {adicionales}"
            )

    @monitor_rendimiento
    def transformar_columna(
        self,
        columna: str,
        transformacion: Union[str, Callable],
        nueva_columna: Optional[str] = None
    ) -> Optional[DataFrame]:
        """
        Aplica una transformación a una columna con verificación de tipo y manejo de errores.
        
        Args:
            columna: Columna a transformar
            transformacion: Expresión SQL o transformación callable
            nueva_columna: Nombre opcional para la nueva columna (por defecto: sobrescribir existente)
            
        Returns:
            DataFrame transformado
        """
        try:
            if not self.datos:
                raise ValueError("No hay datos cargados")

            columna_objetivo = nueva_columna or columna
            
            if callable(transformacion):
                # Crear UDF a partir de callable
                self.datos = self.datos.withColumn(
                    columna_objetivo,
                    F.udf(transformacion)(F.col(columna))
                )
            else:
                # Aplicar expresión SQL
                self.datos = self.datos.withColumn(
                    columna_objetivo,
                    F.expr(transformacion)
                )

            return self.datos

        except Exception as e:
            self.logger.error(f"Error al transformar la columna {columna}: {str(e)}")
            raise

    def obtener_informe_rendimiento(self) -> Dict[str, Any]:
        """Genera un informe completo de rendimiento."""
        if not self._metricas_rendimiento:
            return {"mensaje": "No hay métricas de rendimiento disponibles"}

        metricas_recientes = self._metricas_rendimiento[-1]
        return {
            "estado_actual": {
                "filas": metricas_recientes.cantidad_filas,
                "columnas": metricas_recientes.cantidad_columnas,
                "uso_memoria": metricas_recientes.uso_memoria,
                "esquema": metricas_recientes.esquema
            },
            "historial_rendimiento": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "tiempo_ejecucion": m.tiempo_ejecucion,
                    "uso_memoria": m.uso_memoria
                }
                for m in self._metricas_rendimiento
            ],
            "analisis_nulos": metricas_recientes.conteo_nulos
        }

    def __enter__(self):
        """Entrada del administrador de contexto."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Salida del administrador de contexto con limpieza adecuada."""
        if hasattr(self, 'spark') and self.spark:
            self.spark.stop()
            self.logger.info("SparkSession cerrada exitosamente")

# Ejemplo de uso:
if __name__ == "__main__":
    # Crear manager con configuración personalizada
    gestor = GestorDatosSpark(nombre_app="AnalisisDatos", nivel_log=logging.DEBUG)
    
    # Definir esquema para validación de datos
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType
    
    esquema = StructType([
        StructField("id", IntegerType(), False),
        StructField("nombre", StringType(), True),
        StructField("valor", IntegerType(), True)
    ])
    
    # Leer y procesar datos con monitoreo de rendimiento
    try:
        with gestor as gds:
            df = gds.leer_archivo(
                "data.csv",
                esquema=esquema,
                validar_esquema=True
            )
            
            # Aplicar transformación
            df = gds.transformar_columna(
                "valor",
                "CASE WHEN valor IS NULL THEN 0 ELSE valor * 2 END",
                "valor_doblado"
            )
            
            # Obtener métricas de rendimiento
            informe_rendimiento = gds.obtener_informe_rendimiento()
            print("Informe de Rendimiento:", informe_rendimiento)
            
    except Exception as e:
        logging.error(f"Error en el procesamiento de datos: {str(e)}")
