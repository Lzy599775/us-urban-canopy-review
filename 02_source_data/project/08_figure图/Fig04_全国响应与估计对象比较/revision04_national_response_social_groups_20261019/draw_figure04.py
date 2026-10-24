"""Render the approved eight-panel Figure 4 from frozen results; no model fitting."""
from nc_paths import project_root as _nc_project_root, external_path as _nc_external_path
from pathlib import Path
import hashlib
import json
import sys

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
import pymupdf

sys.stdout.reconfigure(encoding='utf-8')
ROOT = _nc_project_root()
OUT = Path(__file__).resolve().parent
PLAN = OUT.parent / 'plan_20261019/qa/plan_source_check.json'
BASEMAP = OUT.parent / 'revision03_atlas45/prepared/locator_states.gpkg'
NAME = 'Figure04_national_response_and_social_groups'
MM = (180, 235)
EXT = (-2450000, 2350000, 200000, 3270000)
INK = '#2D3940'
QCOL = ['#70C1AB', '#EAB77D', '#AAA5CF', '#D98EB7']
QMARK = ['o', 's', '^', 'D']
DIMS = [('income', 'Income'), ('poverty', 'Poverty'), ('age65plus', 'Older population')]
NORM = Normalize(-.55, .05, clip=False)
CMAP = LinearSegmentedColormap.from_list('linear_response', [
    (0., '#4E9291'), ((-.15 + .55)/.6, '#B5D5CF'),
    ((0 + .55)/.6, '#F6F4EE'), (1., '#D8A09A')], N=1024)
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': 7,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.linewidth': .5,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'axes.unicode_minus': True, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def equal(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)


def label(fig, x, y, text, size=7, weight='normal', ha='left'):
    return fig.text(x/MM[0], y/MM[1], text, fontsize=size, fontweight=weight,
                    ha=ha, va='center', color=INK)


def axis(fig, x, y, w, h, gid):
    a = fig.add_axes([x/MM[0], y/MM[1], w/MM[0], h/MM[1]])
    a.set_gid(gid)
    return a


def title(fig, x, y, letter, text):
    label(fig, x, y, letter, 10, 'bold')
    label(fig, x+5, y, text, 8)


def forest_axis(a, xlim, ylim, xticks):
    a.set(xlim=xlim, ylim=ylim, xticks=xticks, yticks=[])
    a.spines[['left', 'right', 'top']].set_visible(False)
    a.spines['bottom'].set_color('#929DA1')
    a.tick_params(axis='x', labelsize=7, width=.45, length=2, pad=3, colors=INK)
    a.axvline(0, color='#A6B0B3', lw=.5, ls=(0, (2, 2)), zorder=0)


def point_interval(a, est, low, high, y, color, marker, gid, filled=True):
    assert a.get_xlim()[0] < est < a.get_xlim()[1]
    if np.isfinite(low) and np.isfinite(high):
        assert a.get_xlim()[0] < low <= est <= high < a.get_xlim()[1]
        line, = a.plot([low, high], [y, y], color=color, lw=.9,
                       solid_capstyle='round', zorder=2)
        line.set_gid(gid+'_interval')
        equal(line.get_xdata(), [low, high])
        for j, v in enumerate([low, high]):
            cap, = a.plot([v, v], [y-.045, y+.045], color=color, lw=.7, zorder=2)
            cap.set_gid(gid+'_cap_'+str(j))
    else:
        assert np.isnan(low) and np.isnan(high)
    dot, = a.plot(est, y, marker=marker, ms=4.3, linestyle='none',
                  mfc=color if filled else 'white', mec=color, mew=.8, zorder=3)
    dot.set_gid(gid+'_point')
    equal(dot.get_xdata(), [est])


def prepare():
    for sub in ['exports', 'source_data', 'qa']:
        (OUT/sub).mkdir(exist_ok=True)
    check = json.loads(PLAN.read_text('utf-8'))
    assert check['status'] == 'PASS_FOR_PLAN'
    sources = check['sources']
    for item in sources.values():
        assert sha(item['path']) == item['sha256'], ('Source changed', item['path'])
    csvopts = dict(float_precision='round_trip')
    scores = pd.read_csv(sources['predictions']['path'], dtype={'state': str, 'tract_id': str}, **csvopts)
    locations = pd.read_csv(sources['locations']['path'], dtype={'state': str, 'tract_id': str}, **csvopts)
    groups = pd.read_csv(sources['groups']['path'], **csvopts)
    formal = pd.read_csv(sources['groups_original']['path'], **csvopts)
    targets = pd.read_csv(sources['targets']['path'], **csvopts)
    pd.testing.assert_frame_equal(groups, formal, check_exact=True)
    keys = ['state', 'tract_id']
    assert len(scores) == 104371 and len(locations) == 42373
    assert not scores.duplicated(keys+['outcome_year']).any()
    assert scores.groupby(keys).theta.nunique().eq(1).all()
    assert scores[keys+['theta', 'orthogonal_score']].notna().all().all()
    assert not locations.duplicated(keys).any()
    points = scores[keys+['theta']].drop_duplicates().merge(
        locations[keys+['display_x_epsg5070_m', 'display_y_epsg5070_m', 'analysis_records']],
        on=keys, how='outer', validate='one_to_one', indicator=True)
    assert len(points) == 42373 and points._merge.eq('both').all()
    points = points.drop(columns='_merge').sort_values(keys).reset_index(drop=True)
    counts = scores.groupby(keys).size().rename('actual_records').reset_index()
    matched = points.merge(counts, on=keys, validate='one_to_one')
    assert matched.analysis_records.eq(matched.actual_records).all()
    assert points.notna().all().all() and points.theta.between(-.55, .05).all()
    assert points.display_x_epsg5070_m.between(*EXT[:2]).all()
    assert points.display_y_epsg5070_m.between(*EXT[2:]).all()
    assert int(points.theta.gt(0).sum()) == 20 and int(points.theta.eq(0).sum()) == 0

    names = [
        ('Geographic adjustment', '地理干扰变量调整后的广义随机森林'),
        ('Population weighting', '人口规模加权地理调整正交得分'),
        ('Tract fixed effects', '小区固定效应比较模型'),
        ('Historical sample', '未加地理控制的广义随机森林'),
    ]
    selected = []
    for label_en, target in names:
        z = targets[targets['估计对象'].eq(target)]
        z = z[z['空间尺度_公里'].isna() if label_en == 'Historical sample' else z['空间尺度_公里'].eq(200)]
        assert len(z) == 1
        row = z.iloc[0]
        selected.append(dict(label=label_en, source_target=target,
                             estimate=float(row['点估计_摄氏度每树冠百分点']),
                             lower95=float(row['条件区间下限95']), upper95=float(row['条件区间上限95']),
                             records=int(row['记录数']), tracts=int(row['小区数']),
                             bandwidth_km=float(row['空间尺度_公里']),
                             weight=row['权重'], control=row['地理控制']))
    national = pd.DataFrame(selected)
    equal(national.estimate.iloc[0], scores.orthogonal_score.mean())
    assert national.records.iloc[:3].eq(104371).all()
    assert national.tracts.iloc[:3].eq(42373).all()
    assert national.records.iloc[3] == 105437 and national.tracts.iloc[3] == 42805
    means = groups[groups.comparison.eq('group_mean') & groups.bandwidth_km.eq(200)].copy()
    contrasts = groups[groups.comparison.eq('q4_minus_q1')].copy()
    assert len(groups) == 45 and len(means) == 12 and len(contrasts) == 9
    assert not means.duplicated(['dimension', 'group']).any()
    assert not contrasts.duplicated(['dimension', 'bandwidth_km']).any()
    for dimension, _ in DIMS:
        m = means[means.dimension.eq(dimension)]
        assert set(m.group) == {1, 2, 3, 4}
        assert m.rows.sum() == 103656 and m.tracts.sum() == 42082
        c = contrasts[contrasts.dimension.eq(dimension)]
        assert set(c.bandwidth_km) == {100, 200, 400}
        for km in [100, 200, 400]:
            g = groups[groups.dimension.eq(dimension) & groups.comparison.eq('group_mean') & groups.bandwidth_km.eq(km)].set_index('group')
            equal(g.loc[4, 'estimate']-g.loc[1, 'estimate'], c[c.bandwidth_km.eq(km)].estimate.iloc[0])
    assert (means.estimate < 0).all()
    assert contrasts[contrasts.dimension.eq('income')].holm_p_three_contrasts.gt(.05).all()
    pov = contrasts[contrasts.dimension.eq('poverty')].set_index('bandwidth_km')
    assert pov.loc[100, 'holm_p_three_contrasts'] > .05
    assert pov.loc[[200, 400], 'holm_p_three_contrasts'].lt(.05).all()
    age = contrasts[contrasts.dimension.eq('age65plus')]
    assert (age.lower95.lt(0) & age.upper95.gt(0)).all()

    states = gpd.read_file(BASEMAP)
    assert len(states) == 49 and states.crs.to_epsg() == 5070
    assert states.is_valid.all() and not states.geometry.is_empty.any()
    xmin, ymin, xmax, ymax = states.total_bounds
    assert EXT[0] <= xmin < xmax <= EXT[1] and EXT[2] <= ymin < ymax <= EXT[3]
    sources['basemap'] = dict(path=str(BASEMAP), sha256=sha(BASEMAP))
    for data, filename in [(points, 'map_predictions.csv'), (national, 'national_estimates.csv'),
                           (means, 'group_means_200km.csv'), (contrasts, 'group_contrasts_all_scales.csv')]:
        data.to_csv(OUT/'source_data'/filename, index=False, encoding='utf-8-sig')
    receipt = dict(source_files=sources, input_records=len(scores), unique_tracts=len(points),
                   duplicate_tract_year_keys=0, unmatched_locations=0, missing_map_values=0,
                   repeated_predictions_identical=True, positive_predictions=20, zero_predictions=0,
                   prediction_range=[float(points.theta.min()), float(points.theta.max())],
                   record_weighted_prediction_mean=float(scores.theta.mean()),
                   record_weighted_score_mean=float(scores.orthogonal_score.mean()),
                   grouped_records_per_dimension=103656, grouped_tracts_per_dimension=42082,
                   unassigned_records=715, unassigned_tracts=291,
                   group_source_exact_match=True, group_contrast_identity=True,
                   basemap_crs='EPSG:5070', basemap_valid=True, basemap_extent=states.total_bounds.tolist(),
                   coordinate_extent=list(EXT), formal_model_run=False, new_inference=False,
                   prior_figures_modified=False)
    return points, national, means, contrasts, states, receipt


def map_decor(a):
    forward = Transformer.from_crs(4326, 5070, always_xy=True)
    inverse = Transformer.from_crs(5070, 4326, always_xy=True)
    geod = Geod(ellps='GRS80')
    x0 = EXT[0]+.05*(EXT[1]-EXT[0])
    y0 = EXT[2]+.075*(EXT[3]-EXT[2])
    def distance(dx):
        return geod.inv(*inverse.transform(x0, y0), *inverse.transform(x0+dx, y0))[2]
    def projected_length(target):
        lo, hi = 0., 2000000.
        for _ in range(65):
            mid = (lo+hi)/2
            if distance(mid) < target:
                lo = mid
            else:
                hi = mid
        result = (lo+hi)/2
        assert abs(distance(result)-target) < .01
        return result
    lengths = [0., projected_length(500000), projected_length(1000000)]
    a.plot([x0, x0+lengths[-1]], [y0, y0], color=INK, lw=.65, zorder=5)
    for dx, text in zip(lengths, ['0', '500', '1,000 km']):
        a.plot([x0+dx]*2, [y0-18000, y0+18000], color=INK, lw=.55, zorder=5)
        a.annotate(text, (x0+dx, y0), xytext=(0, -4), textcoords='offset points',
                   ha='center', va='top', fontsize=7, color=INK, zorder=5)
    nx, ny = forward.transform(-96, 48.8)
    lon, lat, _ = geod.fwd(-96, 48.8, 0, 1000)
    xx, yy = forward.transform(lon, lat)
    assert abs(xx-nx) < 1e-6 and yy > ny
    fx, fy = a.transAxes.inverted().transform(a.transData.transform([nx, ny]))
    a.plot([fx, fx], [fy-.05, fy+.01], transform=a.transAxes, color=INK, lw=.6, zorder=5)
    a.add_patch(Polygon([[fx, fy+.025], [fx-.006, fy+.006], [fx+.006, fy+.006]],
                        transform=a.transAxes, facecolor=INK, edgecolor='none', zorder=5))
    a.text(fx, fy+.037, 'N', transform=a.transAxes, fontsize=7, ha='center', va='bottom', color=INK)
    return dict(ellipsoid='GRS80', projected_start=[x0, y0], projected_lengths_m=lengths,
                actual_distances_m=[distance(dx) for dx in lengths],
                north_reference_lonlat=[-96, 48.8], north_projected_dx_m=float(xx-nx))


def draw(points, national, means, contrasts, states, rec):
    font_manager.findfont('Arial', fallback_to_default=False)
    f = plt.figure(figsize=(MM[0]/25.4, MM[1]/25.4), dpi=180)
    title(f, 4, 231, 'a', 'Predicted response')
    a = axis(f, 4, 158, 108, 108*(EXT[3]-EXT[2])/(EXT[1]-EXT[0]), 'panel_a')
    states.plot(ax=a, facecolor='white', edgecolor='#CCD2D3', linewidth=.25, aspect=None)
    # Union is a display outline only; the source geometries are never altered.
    gpd.GeoSeries([states.geometry.union_all()], crs=states.crs).boundary.plot(
        ax=a, color='#8D999D', linewidth=.45, aspect=None)
    dots = a.scatter(points.display_x_epsg5070_m, points.display_y_epsg5070_m,
                     c=points.theta, s=.55, cmap=CMAP, norm=NORM,
                     linewidths=0, edgecolors='none', alpha=1, zorder=3)
    dots.set_gid('national_42373_predictions')
    equal(dots.get_array(), points.theta)
    equal(dots.get_offsets(), points[['display_x_epsg5070_m', 'display_y_epsg5070_m']])
    a.set(xlim=EXT[:2], ylim=EXT[2:], xticks=[], yticks=[])
    a.set_aspect('equal')
    a.xaxis.offsetText.set_visible(False)
    a.yaxis.offsetText.set_visible(False)
    a.set_axis_off()
    rec['cartography'] = map_decor(a)
    ca = axis(f, 9, 149, 98, 2.6, 'response_colourbar')
    cb = f.colorbar(matplotlib.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=ca,
                    orientation='horizontal', ticks=[-.5, -.4, -.3, -.2, -.1, 0, .05])
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor('face')
    cb.outline.set_linewidth(.35)
    cb.ax.tick_params(labelsize=7, width=.4, length=2, pad=2)
    cb.ax.set_xticklabels(['−0.5', '−0.4', '−0.3', '−0.2', '−0.1', '0', '0.05'])
    cb.ax.axvline(0, color='#80898B', lw=.5)
    label(f, 58, 154, 'Response (°C per pp)', 7.5, ha='center')

    title(f, 124, 231, 'b', 'National estimates')
    b = axis(f, 124, 158, 50, 65, 'panel_b')
    forest_axis(b, (-.26, .02), (-.42, 3.7), [-.2, -.1, 0])
    colors = ['#4E9291', '#83ABC1', '#A295BF', '#8F989C']
    for i, row in national.iterrows():
        y = [3., 2., 1., -.02][i]
        b.text(0, y+.35, row.label, transform=b.get_yaxis_transform(), fontsize=7.5, color=INK)
        point_interval(b, row.estimate, row.lower95, row.upper95, y, colors[i],
                       ['o', 's', 'D', 'o'][i], f'national_{i}', filled=i < 3)
    label(f, 149, 149, 'Response (°C per pp)', 7.5, ha='center')

    label(f, 4, 140, 'Group means · 200 km', 8, 'bold')
    for j, (dim, heading) in enumerate(DIMS):
        x = 4+59*j
        letter = 'cde'[j]
        title(f, x, 133, letter, heading)
        ax = axis(f, x+7, 90, 33, 34, 'panel_'+letter)
        forest_axis(ax, (-.26, .02), (.5, 4.5), [-.2, -.1, 0])
        label(f, x+50, 127, 'n', 7, ha='center')
        for q in [1, 2, 3, 4]:
            row = means[means.dimension.eq(dim) & means.group.eq(q)].iloc[0]
            y = 5-q
            point_interval(ax, row.estimate, row.lower95, row.upper95, y,
                           QCOL[q-1], QMARK[q-1], f'{dim}_Q{q}')
            yy = 90+(y-.5)/4*34
            label(f, x+.2, yy, f'Q{q}', 7)
            label(f, x+50, yy, f'{int(row.rows):,}', 7, ha='center')
    label(f, 90, 82, 'Mean conditional response (°C per pp)', 7.5, ha='center')

    label(f, 4, 74, 'Q4 − Q1', 8, 'bold')
    for j, (dim, heading) in enumerate(DIMS):
        x = 4+59*j
        letter = 'fgh'[j]
        title(f, x, 64, letter, heading)
        ax = axis(f, x+9, 22, 30, 32, 'panel_'+letter)
        forest_axis(ax, (-.1, .1), (.5, 3.5), [-.1, 0, .1])
        label(f, x+2.7, 58.5, 'km', 7, ha='center')
        label(f, x+49.5, 58.5, 'Holm p', 7, ha='center')
        for i, km in enumerate([100, 200, 400]):
            row = contrasts[contrasts.dimension.eq(dim) & contrasts.bandwidth_km.eq(km)].iloc[0]
            y = 3-i
            point_interval(ax, row.estimate, row.lower95, row.upper95, y,
                           '#647F8C', 'o', f'{dim}_{km}km', filled=km == 200)
            yy = 22+(y-.5)/3*32
            label(f, x+2.7, yy, str(km), 7, ha='center')
            label(f, x+49.5, yy, f'{row.holm_p_three_contrasts:.4f}', 7, ha='center')
    label(f, 90, 12, 'Response difference (°C per pp)', 7.5, ha='center')
    rec.update(figure_size_mm=list(MM), point_area_pt2=.55, point_order='state, tract_id',
               map_color_limits=[-.55, .05], color_normalization='linear',
               color_nodes=[[-.55, '#4E9291'], [-.15, '#B5D5CF'], [0, '#F6F4EE'], [.05, '#D8A09A']],
               map_values_clipped=0, group_colors=QCOL, group_intervals='200 km, existing pointwise 95% reference intervals',
               contrast_scales_km=[100, 200, 400], p_adjustment='existing Holm within three contrasts at each scale',
               mean_axis_limits=[-.26, .02], contrast_axis_limits=[-.1, .1])
    return f


def export(f, rec):
    f.canvas.draw()
    renderer = f.canvas.get_renderer()
    texts, clipped, overlaps = [], [], []
    for t in f.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            box = t.get_window_extent(renderer)
            texts.append((t, box))
            if box.x0 < -.5 or box.y0 < -.5 or box.x1 > f.bbox.x1+.5 or box.y1 > f.bbox.y1+.5:
                clipped.append(t.get_text())
    for i, (t, box) in enumerate(texts):
        for u, other in texts[i+1:]:
            w = min(box.x1, other.x1)-max(box.x0, other.x0)
            h = min(box.y1, other.y1)-max(box.y0, other.y0)
            if w > f.dpi/72*.2 and h > f.dpi/72*.2:
                overlaps.append([t.get_text(), u.get_text(), float(w), float(h)])
    layout = dict(text_count=len(texts), minimum_text_pt=min(t.get_fontsize() for t, _ in texts),
                  text_outside_canvas=clipped, overlapping_text=overlaps)
    (OUT/'qa/layout_check.json').write_text(json.dumps(layout, ensure_ascii=False, indent=2), 'utf-8')
    assert not clipped, ('Text outside canvas', clipped)
    assert not overlaps, ('Overlapping text', overlaps)
    assert layout['minimum_text_pt'] >= 7
    for ext in ['pdf', 'svg', 'png']:
        f.savefig(OUT/'exports'/f'{NAME}.{ext}', dpi=600, facecolor='white')
    with pymupdf.open(OUT/'exports'/f'{NAME}.pdf') as pdf:
        pdf[0].get_pixmap(matrix=pymupdf.Matrix(1.8, 1.8)).save(OUT/'qa'/f'{NAME}_preview.png')
    plt.close(f)
    for item in rec['source_files'].values():
        assert sha(item['path']) == item['sha256'], ('Source changed during drawing', item['path'])
    rec.update(status='PASS', layout=layout, export_dpi=600,
               output_files={ext: {'path': str(OUT/'exports'/f'{NAME}.{ext}'),
                                   'sha256': sha(OUT/'exports'/f'{NAME}.{ext}')}
                             for ext in ['pdf', 'svg', 'png']})
    (OUT/'qa/drawing_receipt.json').write_text(json.dumps(rec, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps(dict(status=rec['status'], points=rec['unique_tracts'], layout=layout,
                          outputs=[v['path'] for v in rec['output_files'].values()]), ensure_ascii=False))


if __name__ == '__main__':
    points, national, means, contrasts, states, receipt = prepare()
    figure = draw(points, national, means, contrasts, states, receipt)
    export(figure, receipt)
