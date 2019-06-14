<!DOCTYPE html>
<html lang="en">
<style>
body {
    background-image: url("science_bck.jpg");
    background-repeat: no-repeat;
    background-position: right top;
    background-attachment: fixed;
    background-size: 100%;
    text-align:center;
}

div {
    position: relative;
    -moz-box-shadow: 1px 2px 4px rgba(0, 0, 0,0.5);
    -webkit-box-shadow: 1px 2px 4px rgba(0, 0, 0, .5);
    box-shadow: 1px 2px 4px rgba(0, 0, 0, .5);
    padding: 20px;
    background: white;
    display: inline-block;
}
</style>
<head>
    <meta charset="UTF-8">
    <title>DAFD</title>
</head>
<body>
<div>
    <img src="DAFD_logo.png" alt="DAFD" style="width:600px;height:300px;">
</div>

<br>
<br>

<?php

$orifice_size=$_POST['orifice-size'];
$aspect_ratio=$_POST['aspect-ratio'];
$expansion_ratio=$_POST['expansion-ratio'];
$normalized_orifice_length=$_POST['normalized-orifice-length'];
$normalized_water_inlet=$_POST['normalized-water-inlet'];
$normalized_oil_inlet=$_POST['normalized-oil-inlet'];
$flow_rate_ratio=$_POST['flow-rate-ratio'];
$capillary_number=$_POST['capillary-number'];

$constraints = array($orifice_size, $aspect_ratio, $expansion_ratio, $normalized_orifice_length, $normalized_water_inlet, $normalized_oil_inlet, $flow_rate_ratio, $capillary_number);
$constraint_names = array("orifice_size", "aspect_ratio", "expansion_ratio", "normalized_orifice_length",
			"normalized_water_inlet", "normalized_oil_inlet", "flow_rate_ratio", "capillary_number");

$DAFD_location = "/home/chris/Work/DAFD/DAFD/";
$file = $DAFD_location . "cmd_inputs.txt";
file_put_contents($file, "FORWARD\n");

for($i=0; $i<sizeof($constraints); $i++)
{
	$value = $constraints[$i];
	$current = $constraint_names[$i] . "=" . $value . "\n";
	file_put_contents($file, $current, FILE_APPEND);
}


$outputs = shell_exec($DAFD_location . "venv/bin/python3 " . $DAFD_location . "DAFD_CMD.py");
$arr_outs = explode("|",explode("BEGIN:",$outputs)[1]);
?>

<div>
    <h1>Forward Model Inputs</h1>

    Orifice Size:
    <?php
    echo $orifice_size;
    ?>
    <br>

    Aspect Ratio:
    <?php
    echo $aspect_ratio;
    ?>
    <br>

    Expansion Ratio:
    <?php
    echo $expansion_ratio;
    ?>
    <br>

    Normalized Orifice Length:
    <?php
    echo $normalized_orifice_length;
    ?>
    <br>

    Normalized Water Inlet:
    <?php
    echo $normalized_water_inlet;
    ?>
    <br>

    Normalized Oil Inlet:
    <?php
    echo $normalized_oil_inlet;
    ?>
    <br>

    Flow Rate Ratio:
    <?php
    echo $flow_rate_ratio;
    ?>
    <br>

    Capillary Number:
    <?php
    echo $capillary_number;
    ?>
    <br>

</div>

<br>
<br>

<div>
    <h1>Forward Model Results</h1>

    Generation Rate:
    <?php
    echo $arr_outs[0];
    ?>
    <br>

    Droplet Size:
    <?php
    echo $arr_outs[1];
    ?>
    <br>

    Regime:
    <?php
    echo $arr_outs[2];
    ?>
    <br>

</div>

<br>
<br>

<div>
    <h1>Calculated Values</h1>

    Oil Flow Rate (ml/hr):
    <?php
    echo $arr_outs[3];
    ?>
    <br>

    Water Flow Rate (ul/min):
    <?php
    echo $arr_outs[4];
    ?>
    <br>

    Droplet Inferred Size:
    <?php
    echo $arr_outs[5];
    ?>
    <br>

</div>

</body>
