"""
Módulo: map_builder.py
Responsabilidad: Visualización interactiva y geoespacial del riesgo por MLOps.

Utiliza Folium para renderizar la telemetría en un HTML estático (Capa de Presentación).
"""

import logging
from pathlib import Path

import pandas as pd
import folium
from branca.element import Template, MacroElement

logger = logging.getLogger(__name__)

class SupplyChainMap:
    """
    Generador de Mapas de Riesgo de Cadena de Suministro (MVP).
    Inyecta data de NLP+Survival Model en visuales geográficas HTML.
    """

    # Diccionario maestro de coordenadas (Puertos, Chokepoints y Macroregiones)
    PORT_COORDINATES = {
        # Tier-1 Ports (Hubs globales)
        "shanghai": [31.2304, 121.4737],
        "singapore": [1.3521, 103.8198],
        "rotterdam": [51.9225, 4.4791],
        "los angeles": [33.7288, -118.2620],
        "new york": [40.7128, -74.0060],
        "hamburg": [53.5511, 9.9937],
        "valencia": [39.4699, -0.3763],
        "jebel ali": [24.9857, 55.0611],
        "busan": [35.1796, 129.0756],
        "antwerp": [51.2194, 4.4025],
        
        # Tier-2 Ports originados en mocks
        "shenzhen": [22.5431, 114.0579],
        "ningbo": [29.8683, 121.5440],
        "long beach": [33.7701, -118.1937],
        "felixstowe": [51.9634, 1.3503],
        "mumbai": [18.9438, 72.8353],
        "colombo": [6.9271, 79.8612],
        "durban": [-29.8587, 31.0218],
        "jeddah": [21.4858, 39.1925],

        # Chokepoints (Puntos de Estrangulamiento Geopolítico)
        "suez canal": [30.5852, 32.2654],
        "suez": [30.5852, 32.2654],
        "panama canal": [9.1012, -79.6800],
        "panama": [9.1012, -79.6800],
        "taiwan": [24.8066, 119.9234],
        "taiwan strait": [24.8066, 119.9234],
        "hormuz": [26.5667, 56.2500],
        "strait of hormuz": [26.5667, 56.2500],
        "malacca": [4.0000, 100.0000],
        "strait of malacca": [4.0000, 100.0000],
        "bosphorus": [41.0256, 29.0350],
        "english channel": [50.1873, -0.5360],
        
        # Macroregiones simuladas en el Mock NLP
        "southeast asia": [15.0, 105.0],
        "east china": [30.0, 122.0],
        "northern europe": [55.0, 10.0],
        "south america": [-15.0, -60.0],
        "west africa": [10.0, 0.0],
        "east africa": [-3.0, 38.0],
        "indian subcontinent": [20.0, 77.0],
        "middle east": [25.0, 45.0],
    }

    def __init__(self, master_table: pd.DataFrame, news_enriched: pd.DataFrame):
        self.master_table = master_table
        self.news_enriched = news_enriched
        # Base tile oscura para contraste con colores de riesgo logístico
        self.m = folium.Map(location=[30.0, 0.0], zoom_start=2, tiles='CartoDB dark_matter')
        
        self.routes_drawn = 0
        self.alerts_placed = 0

    def _get_coords(self, location_name: str) -> list:
        # Recupera las coordenadas de nuestro diccionario de inteligencia espacial
        clean_name = str(location_name).strip().lower()
        if clean_name in self.PORT_COORDINATES:
            return self.PORT_COORDINATES[clean_name]
        logger.warning("Coordenadas no definidas para %s -> Forzando a Nulo (0, 0)", location_name)
        return [0.0, 0.0]

    def _generate_overlay(self, global_score: float) -> str:
        """Dashboard overlay interactivo flotante."""
        template = """
        {% macro html(this, kwargs) %}
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 260px; height: 120px; 
                    background-color: white; border:2px solid gray; z-index:9999;
                    font-size:14px; padding: 10px; border-radius: 8px; font-family: Arial;">
            <b style="color: #333;">🌍 Global Resilience Score</b><br>
            <span style="font-size: 32px; font-weight: bold; color: {% if score > 75 %}green{% elif score > 40 %}orange{% else %}red{% endif %};">
                {{ "{:.1f}".format(score) }} / 100
            </span><br>
            <i style="color: #666; font-size: 11px;">Powered by GSCRI MLOps</i><br>
            <i style="color: #666; font-size: 11px;">(Higher is better / safer)</i>
        </div>
        {% endmacro %}
        """
        return template

    def build_map(self) -> folium.Map:
        """Orquesta las capas del raster geográfico y calcula agregados globales."""
        logger.info("Comenzando motor de renderizado geométrico Folium.")
        
        # ── 1. Capa de Rutas Marítimas ─────────────────────────────────
        # Filtrar solo barcos activos ("In Transit") para una foto Snapshot
        active_ships = self.master_table[self.master_table["status"].str.lower() == "in transit"]
        
        for _, row in active_ships.iterrows():
            route_id = str(row.get("route_id", ""))
            if "-" not in route_id:
                continue
            origin, dest = route_id.split("-", 1)
            
            orig_coords = self._get_coords(origin)
            dest_coords = self._get_coords(dest)
            
            # Avoid breaking line if both are [0, 0] or unknown
            if orig_coords == [0.0, 0.0] and dest_coords == [0.0, 0.0]:
                continue
                
            risk = float(row.get("voyage_risk_index", 0.0))
            
            color = "green"
            weight = 2
            if risk > 0.7:
                color = "#ff3333" # Rojo fuerte
                weight = 4
            elif risk >= 0.5:
                color = "orange"
                weight = 3
                
            tooltip_html = f"<b>Vessel:</b> {row.get('vessel_id')}<br><b>Cargo:</b> {row.get('cargo_type')}<br><b>Risk Score:</b> {risk:.2f}"
            
            folium.PolyLine(
                locations=[orig_coords, dest_coords],
                color=color,
                weight=weight,
                opacity=0.7,
                tooltip=tooltip_html
            ).add_to(self.m)
            
            self.routes_drawn += 1

        # ── 2. Capa de Puntos Calientes (NLP News Alerts) ────────────────
        hotspots_df = self.news_enriched[self.news_enriched["extracted_locations"].notna()]
        
        for _, news in hotspots_df.iterrows():
            locs_raw = str(news.get("extracted_locations", "unknown"))
            if locs_raw.lower() == "unknown":
                continue
                
            risk_score = float(news.get("risk_score", 0.0))
            
            # Icon alert if critical
            icon_name = "info-sign"
            icon_color = "lightgray"
            if risk_score > 0.8:
                icon_name = "warning-sign"
                icon_color = "red"
            elif risk_score >= 0.5:
                # yellow / orange marker
                icon_color = "orange"
            
            locs_list = [l.strip() for l in locs_raw.split(",")]
            for loc in locs_list:
                coords = self._get_coords(loc)
                if coords != [0.0, 0.0]:
                    popup_text = f"<b>Risk:</b> {risk_score:.2f}<br><b>News:</b> {str(news['headline_clean']).title()}"
                    
                    folium.Marker(
                        location=coords,
                        popup=folium.Popup(popup_text, max_width=300),
                        icon=folium.Icon(color=icon_color, icon=icon_name)
                    ).add_to(self.m)
                    
                    self.alerts_placed += 1

        # ── 3. Macro Dashboard (Resilience Score) ─────────────────────
        # Invert average risk to 0-100 score format (Highest risk -> Score drops)
        avg_risk = active_ships["voyage_risk_index"].mean() if not active_ships.empty else 0.5
        global_resilience_score = (1.0 - avg_risk) * 100.0

        macro = MacroElement()
        macro._template = Template(self._generate_overlay(global_resilience_score).replace("score", str(global_resilience_score)))
        self.m.get_root().add_child(macro)
        
        logger.info("Renderizado de mapa exitoso. %d Rutas trazadas | %d Alertas de prensa colocadas.", self.routes_drawn, self.alerts_placed)
        return self.m

    def save(self, filepath: Path) -> None:
        """Guarda permanentemente el output generado."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.m.save(str(filepath))
        logger.info("✔️ Mapa web exportado como: %s", filepath)
