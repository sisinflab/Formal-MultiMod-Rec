# A Formal Multimodal Framework for Personalized Multimedia Recommendation

A table of multimodal recommender systems publications. This page will be ****periodically**** updated to include recent works.

The table is taken from **A Formal Multimodal Framework for Personalized Multimedia Recommendation** submitted at IEEE Transactions on Multimedia.



## Review

::: table*
::: adjustbox
width=, center

::: tabular
llcccccccccc **Papers** & & & &\
(rl)3-5(lr)6-10(lr)11-12

& & & & & & & & &\
(rl)7-8(rl)9-10 & & & & & & & & & & &\

@DBLP:conf/mm/FerracaniPBMB15 & 2015 & & & & & & & & &\
@DBLP:conf/bigdataconf/JiaWLXXZ15 & 2015 & & & & & & & &\
@DBLP:journals/mta/LiPGCZ15 & 2015 & & & & & & & &\
@DBLP:journals/mta/NieLZS16 & 2016 & & & & & & & & &\
@DBLP:conf/mm/ChenHK16 & 2016 & & & & & & & &\
@DBLP:conf/mm/HanWJD17 & 2017 & & & & & & & & & &\
@DBLP:conf/recsys/OramasNSS17 & 2017 & & & & & & & & & &\
@DBLP:conf/ijcai/ZhangWHHG17 & 2017 & & & & & & & & & &\
@DBLP:conf/kdd/YingHCEHL18 & 2018 & & & & & & & & &\
@DBLP:conf/emnlp/WangNL18 & 2018 & & & & & & & & &\
@DBLP:conf/mm/LiuCSWNK19 & 2019 & & & & & & & & &\
@DBLP:conf/sigir/ChenCXZ0QZ19 & 2019 & & & & & & & &\
@DBLP:conf/mm/WeiWN0HC19 & 2019 & & & & & & & & & &\
@DBLP:journals/tois/ChengCZKK19 & 2019 & & & & & & & &\
@DBLP:conf/mm/DongSFJXN19 & 2019 & & & & & & & & & &\
@DBLP:conf/kdd/ChenHXGGSLPZZ19 & 2019 & & & & & & & & &\
@DBLP:conf/mm/YuSZZJ19 & 2019 & & & & & & & & & &\
@DBLP:journals/tkde/CuiWLZW20 & 2020 & & & & & & & & & &\
@DBLP:conf/mm/WeiWN0C20 & 2020 & & & & & & & & & &\
@DBLP:conf/cikm/SunCZWZZWZ20 & 2020 & & & & & & & & &\
@DBLP:conf/ijcai/Chen020 & 2020 & & & & & & & & &\
@DBLP:journals/tmm/MinJJ20 & 2020 & & & & & & & & &\
@DBLP:conf/ijcnn/Shen0LWC20 & 2020 & & & & & & & & & &\
@DBLP:conf/aaai/YangDW20 & 2020 & & & & & & & & & &\
@DBLP:journals/ipm/TaoWWHHC20 & 2020 & & & & & & & & & &\
@DBLP:journals/tcss/YangWLLGDW20 & 2020 & & & & & & & & & &\
@DBLP:journals/tmm/SangXQMLW21 & 2021 & & & & & & & & & &\
@DBLP:conf/mm/LiuYLWTZSM21 & 2021 & & & & & & & & & &\
@DBLP:conf/mm/Zhang00WWW21 & 2021 & & & & & & & & & &\
@DBLP:conf/bigmm/VaswaniAA21 & 2021 & & & & & & & & & &\
@DBLP:journals/eswa/LeiHZSZ21 & 2021 & & & & & & & & &\
@DBLP:journals/tomccap/WangDJJSN21 & 2021 & & & & & & & & &\
@DBLP:journals/tmm/ZhanLASDK22 & 2022 & & & & & & & & &\
@DBLP:conf/sigir/WuWQZHX22 & 2022 & & & & & & & & & &\
@DBLP:journals/tmm/YiC22 & 2022 & & & & & & & & & &\
@DBLP:conf/sigir/Yi0OM22 & 2022 & & & & & & & & & &\
@DBLP:conf/mir/LiuMSO022 & 2022 & & & & & & & & & &\
:::
:::

[]{#tab:papers label="tab:papers"}
:::



<table cellspacing="0" border="0">
	<caption>Table 1. Overview of the core questions 
		arise when modelling a multimodal recommender system, as observed in the
literature. HFE: Handcrafted Feature Extraction, TFE: Trainable Feature Extraction, MMR: Multi-Modal Representation.</caption>
	<colgroup width="120"></colgroup>
	<colgroup span="5" width="85"></colgroup>
	<colgroup width="118"></colgroup>
	<colgroup span="2" width="85"></colgroup>
	<colgroup span="2" width="146"></colgroup>
	<colgroup span="2" width="85"></colgroup>
	<tr>
		<td style="border-top: 1px solid #000000; border-bottom: 1px solid #000000; border-left: 1px solid #000000" rowspan=3 height="51" align="center" valign=middle><b>Papers</b></td>
		<td style="border-top: 1px solid #000000; border-bottom: 1px solid #000000" rowspan=3 align="center" valign=middle><b>Year</b></td>
		<td style="border-top: 1px solid #000000; border-bottom: 1px solid #000000" colspan=4 align="center"><b>Modalities (<i>Which?</i>)</b></td>
		<td style="border-top: 1px solid #000000; border-bottom: 1px solid #000000" colspan=5 align="center" valign=middle><b>Feature Elaboration (<i>How?</i>)</b></td>
		<td style="border-top: 1px solid #000000; border-bottom: 1px solid #000000; border-right: 1px solid #000000" colspan=2 align="center"><b>Fusion (<i>When</i>?)</b></td>
		</tr>
	<tr>
		<td style="border-bottom: 1px solid #000000" rowspan=2 align="center" valign=middle><i>Visual</i></td>
		<td style="border-bottom: 1px solid #000000" rowspan=2 align="center" valign=middle><i>Textual</i></td>
		<td style="border-bottom: 1px solid #000000" rowspan=2 align="center" valign=middle><i>Audio</i></td>
		<td style="border-bottom: 1px solid #000000" rowspan=2 align="center" valign=middle><i>Sensory</i></td>
		<td style="border-bottom: 1px solid #000000" rowspan=2 align="center" valign=middle>HFE</td>
		<td style="border-bottom: 1px solid #000000" colspan=2 align="center">TFE</td>
		<td style="border-bottom: 1px solid #000000" colspan=2 align="center" valign=middle>MMR</td>
		<td style="border-bottom: 1px solid #000000" rowspan=2 align="center" valign=middle><i>Early</i></td>
		<td style="border-bottom: 1px solid #000000; border-right: 1px solid #000000" rowspan=2 align="center" valign=middle><i>Late</i></td>
	</tr>
	<tr>
		<td style="border-bottom: 1px solid #000000" align="center"><i>Pretrained</i></td>
		<td style="border-bottom: 1px solid #000000" align="center"><i>End-to-End</i></td>
		<td style="border-bottom: 1px solid #000000" align="center" valign=middle><i>Joint</i></td>
		<td style="border-bottom: 1px solid #000000" align="center" valign=middle><i>Coordinate</i></td>
		</tr>
	<tr>
		<td style="border-left: 1px solid #000000" height="17" align="left"><a href="https://dl.acm.org/doi/10.1145/2733373.2807982">Ferracani et al.</a></td>
		<td align="center" sdval="2015" sdnum="1033;">2015</td>
		<td align="left">&#10004;</td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td style="border-right: 1px solid #000000" align="left"><br></td>
	</tr>
	<tr>
		<td style="border-left: 1px solid #000000" height="17" align="left"><a href="https://ieeexplore.ieee.org/document/7363830">Jia et al.</a></td>
		<td align="center" sdval="2015" sdnum="1033;">2015</td>
		<td align="left">&#10004;</td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td style="border-right: 1px solid #000000" align="left"><br></td>
	</tr>
	<tr>
		<td style="border-left: 1px solid #000000" height="17" align="left"><a href="https://link.springer.com/article/10.1007/s11042-013-1825-x">Li et al.</a></td>
		<td align="center" sdval="2015" sdnum="1033;">2015</td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td style="border-right: 1px solid #000000" align="left"><br></td>
	</tr>
	<tr>
		<td style="border-left: 1px solid #000000" height="17" align="left"><a href="https://link.springer.com/article/10.1007%2Fs11042-014-2339-x">Nie et al.</a></td>
		<td align="center" sdval="2016" sdnum="1033;">2016</td>
		<td align="left">&#10004;</td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left">&#10004;</td>
		<td style="border-right: 1px solid #000000" align="left"><br></td>
	</tr>
	<tr>
		<td style="border-left: 1px solid #000000" height="17" align="left"><a href="https://dl.acm.org/doi/10.1145/2964284.2964291">Chen et al.</a></td>
		<td align="center" sdval="2016" sdnum="1033;">2016</td>
		<td align="left">&#10004;</td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="right"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td style="border-right: 1px solid #000000" align="left"><br></td>
	</tr>
	<tr>
		<td style="border-left: 1px solid #000000" height="17" align="left"><a href="https://dl.acm.org/doi/10.1145/3078971.3080545">Nag et al.</a></td>
		<td align="center" sdval="2017" sdnum="1033;">2017</td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left">&#10004;</td>
		<td style="border-right: 1px solid #000000" align="left"><br></td>
	</tr>
	<tr>
		<td style="border-left: 1px solid #000000" height="17" align="left"><a href="https://dl.acm.org/doi/10.1145/3123266.3123394">Han et al.</a></td>
		<td align="center" sdval="2017" sdnum="1033;">2017</td>
		<td align="left">&#10004;</td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left">&#10004;</td>
		<td style="border-right: 1px solid #000000" align="left"><br></td>
	</tr>
	<tr>
		<td style="border-left: 1px solid #000000" height="17" align="left"><a href="https://dl.acm.org/doi/10.1145/3219819.3219890">Ying et al.</a></td>
		<td align="center" sdval="2018" sdnum="1033;">2018</td>
		<td align="left">&#10004;</td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td style="border-right: 1px solid #000000" align="left"><br></td>
	</tr>
	<tr>
		<td style="border-left: 1px solid #000000" height="17" align="left"><a href="https://doi.org/10.18653/v1/d18-1373">Wang et al.</a></td>
		<td align="center" sdval="2018" sdnum="1033;">2018</td>
		<td align="left">&#10004;</td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td style="border-right: 1px solid #000000" align="left"><br></td>
	</tr>
	<tr>
		<td style="border-left: 1px solid #000000" height="17" align="left"><a href="https://doi.org/10.1145/3343031.3351034">Wei et al.</a></td>
		<td align="center" sdval="2019" sdnum="1033;">2019</td>
		<td align="left">&#10004;</td>
		<td align="left">&#10004;</td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td style="border-right: 1px solid #000000" align="left">&#10004;</td>
	</tr>
	<tr>
		<td style="border-left: 1px solid #000000" height="17" align="left"><a href="https://doi.org/10.1145/3343031.3350905">Dong et al.</a></td>
		<td align="center" sdval="2019" sdnum="1033;">2019</td>
		<td align="left">&#10004;</td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left">&#10004;</td>
		<td style="border-right: 1px solid #000000" align="left"><br></td>
	</tr>
	<tr>
		<td style="border-left: 1px solid #000000" height="17" align="left"><a href="https://doi.org/10.1145/3292500.3330652">Chen et al.</a></td>
		<td align="center" sdval="2019" sdnum="1033;">2019</td>
		<td align="left">&#10004;</td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td style="border-right: 1px solid #000000" align="left"><br></td>
	</tr>
	<tr>
		<td style="border-left: 1px solid #000000" height="17" align="left"><a href="https://doi.org/10.1145/3340531.3411947">Sun et al.</a></td>
		<td align="center" sdval="2020" sdnum="1033;">2020</td>
		<td align="left">&#10004;</td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td style="border-right: 1px solid #000000" align="left"><br></td>
	</tr>
	<tr>
		<td style="border-left: 1px solid #000000" height="17" align="left"><a href="https://doi.org/10.24963/ijcai.2020/339">Chen et al.</a></td>
		<td align="center" sdval="2020" sdnum="1033;">2020</td>
		<td align="left">&#10004;</td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td style="border-right: 1px solid #000000" align="left"><br></td>
	</tr>
	<tr>
		<td style="border-left: 1px solid #000000" height="17" align="left"><a href="https://doi.org/10.1109/TMM.2019.2958761">Min et al.</a></td>
		<td align="center" sdval="2020" sdnum="1033;">2020</td>
		<td align="left">&#10004;</td>
		<td align="left">&#10004;</td>
		<td align="left">&#10004;</td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left">&#10004;</td>
		<td align="left"><br></td>
		<td align="left"><br></td>
		<td style="border-right: 1px solid #000000" align="left"><br></td>
	</tr>
	<tr>
		<td style="border-bottom: 1px solid #000000; border-left: 1px solid #000000" height="17" align="left"><a href="https://aaai.org/ojs/index.php/AAAI/article/view/5362">Yang et al.</a></td>
		<td style="border-bottom: 1px solid #000000" align="center" sdval="2020" sdnum="1033;">2020</td>
		<td style="border-bottom: 1px solid #000000" align="left">&#10004;</td>
		<td style="border-bottom: 1px solid #000000" align="left">&#10004;</td>
		<td style="border-bottom: 1px solid #000000" align="left"><br></td>
		<td style="border-bottom: 1px solid #000000" align="left"><br></td>
		<td style="border-bottom: 1px solid #000000" align="left"><br></td>
		<td style="border-bottom: 1px solid #000000" align="left">&#10004;</td>
		<td style="border-bottom: 1px solid #000000" align="left"><br></td>
		<td style="border-bottom: 1px solid #000000" align="left"><br></td>
		<td style="border-bottom: 1px solid #000000" align="left">&#10004;</td>
		<td style="border-bottom: 1px solid #000000" align="left"><br></td>
		<td style="border-bottom: 1px solid #000000; border-right: 1px solid #000000" align="left">&#10004;</td>
	</tr>
</table>
