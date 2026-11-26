"""Unit tests without Databricks, Streamlit or internet access.

Run: python -m pytest tests/test_monte_carlo.py -q
"""
from pathlib import Path
import ast
import html
import json
import math
import numpy as np

source = Path(__file__).parents[1].joinpath('app.py').read_text(encoding='utf-8')
parsed = ast.parse(source)
names = {'MAP_GEOMETRY_B85', 'monte_carlo_itinerary', '_map_lines', '_merc_lat', '_map_projector', '_offline_svg_route', '_route_map_html'}
selected = []
for node in parsed.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
        selected.append(node)
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
        selected.append(node)
module=ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
namespace={'np':np,'json':json,'math':math,'html':html}
exec(compile(module, '<extracted simulation functions>', 'exec'), namespace)
monte_carlo = namespace['monte_carlo_itinerary']


def test_direct_zero_delay_costs():
    result=monte_carlo([{'flight_minutes':120,'layover_after':0}], [np.zeros(50)],simulations=1000,
                       fare=3500,value_hour=200,meal=250)
    assert result['mean_delay']==0
    assert result['miss_any']==0
    assert result['generalized']==3500
    assert result['planned_minutes']==120


def test_direct_persistent_delay():
    result=monte_carlo([{'flight_minutes':120,'layover_after':0}], [np.full(50,60)],simulations=1000,
                       fare=1000,value_hour=200,meal=50)
    assert result['mean_delay']==60
    assert result['p_arrival_late60']==1
    assert result['mean_outlay']==1050
    assert result['generalized']==1250


def test_connection_miss_mechanism():
    legs=[{'flight_minutes':120,'layover_after':60},{'flight_minutes':100,'layover_after':0}]
    first=np.full(40,60)
    last=np.zeros(40)
    result=monte_carlo(legs,[first,last],min_connections=[45],rebook_wait=180,simulations=1000,
                       fare=0,value_hour=0,transfer=100)
    assert result['miss_any']==1
    assert result['mean_misses']==1
    assert result['mean_delay']==225   # 60 min delayed incoming + 45-60 additional gap + 180 rebook
    assert result['mean_outlay']==300  # transfer (100) + meal (default 200)


def test_connection_caught_with_buffer():
    legs=[{'flight_minutes':120,'layover_after':180},{'flight_minutes':100,'layover_after':0}]
    result=monte_carlo(legs,[np.full(40,60), np.zeros(40)],min_connections=[45],
                       simulations=1000,fare=0,value_hour=0)
    assert result['miss_any']==0
    assert result['mean_delay']==0   # arrives late to stop, but catches planned next flight


def test_quantiles_and_reproducibility():
    legs=[{'flight_minutes':115,'layover_after':70},{'flight_minutes':185,'layover_after':0}]
    pools=[np.array([0,0,30,60,120,300]),np.array([0,0,0,25,45,90])]
    a=monte_carlo(legs,pools,min_connections=[50],simulations=1000,seed=42)
    b=monte_carlo(legs,pools,min_connections=[50],simulations=1000,seed=42)
    assert np.array_equal(a['total_delays'],b['total_delays'])
    assert a['p90_delay']>=a['median_delay']
    assert a['p95_delay']>=a['p90_delay']
    assert a['generalized']>=a['mean_outlay']
    assert 0<=a['miss_any']<=1


def test_validation_rejects_bad_itinerary():
    try:
        monte_carlo([{'flight_minutes':120,'layover_after':20},{'flight_minutes':100,'layover_after':0}],
                    [np.zeros(40),np.zeros(40)],min_connections=[45])
    except ValueError as exc:
        assert 'mínimo' in str(exc)
    else:
        raise AssertionError('expected insufficient connection error')


def test_map_is_offline_and_has_geographic_features():
    route=[{'code':'ATL','lat':33.6407,'lon':-84.4277}, {'code':'DFW','lat':32.8998,'lon':-97.0403}, {'code':'LAX','lat':33.9416,'lon':-118.4085}]
    svg=namespace['_offline_svg_route'](route)
    assert 'ATL' in svg and 'DFW' in svg and 'LAX' in svg
    assert 'animateMotion' in svg and 'flightpath' in svg
    assert 'stroke' in svg and 'mapclip' in svg
    assert 'API KEY' not in svg.upper()
    assert 'http://' not in svg and 'https://' not in svg
    assert '<image ' not in svg


def test_html_result_reveal_no_remote_assets():
    route=[{'code':'ATL','lat':33.64,'lon':-84.42},{'code':'LAX','lat':33.94,'lon':-118.40}]
    r={'mean_delay':35.5,'p90_delay':140,'miss_any':.14,'generalized':4020}
    doc=namespace['_route_map_html'](route,r,[])
    assert 'MONTE CARLO' in doc
    assert 'animation:reveal' in doc
    assert '$4,020' in doc
    assert 'cartocdn' not in doc.lower() and 'api key' not in doc.lower()

