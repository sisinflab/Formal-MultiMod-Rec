# Multimodal Recommendation: Technical Challenges and Applications for Social Good

A table of multimodal recommender systems publications. This page will be ****periodically**** updated to include recent works.

The table is taken from **Multimodal Recommendation: Technical Challenges andApplications for Social Good** submitted at ACM Multimedia 2021 in the Brave New ideas Track.



## Review


<table cellspacing="0" border="0">
	<caption>Overview of the core questions which arise when modelling a multimodal recommender system, as observed in the
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
		<td style="border-top: 1px solid #000000; border-bottom: 1px solid #000000" colspan=4 align="center"><b>Modalities (Which?)</b></td>
		<td style="border-top: 1px solid #000000; border-bottom: 1px solid #000000" colspan=5 align="center" valign=middle><b>Feature Elaboration (How?)</b></td>
		<td style="border-top: 1px solid #000000; border-bottom: 1px solid #000000; border-right: 1px solid #000000" colspan=2 align="center"><b>Fusion (When?)</b></td>
		</tr>
	<tr>
		<td style="border-bottom: 1px solid #000000" rowspan=2 align="center" valign=middle>Visual</td>
		<td style="border-bottom: 1px solid #000000" rowspan=2 align="center" valign=middle>Textual</td>
		<td style="border-bottom: 1px solid #000000" rowspan=2 align="center" valign=middle>Audio</td>
		<td style="border-bottom: 1px solid #000000" rowspan=2 align="center" valign=middle>Sensory</td>
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
