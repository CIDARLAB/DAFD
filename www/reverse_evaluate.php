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

$generation_rate=$_POST['generation-rate'];
$droplet_size=$_POST['droplet-size'];

$constraints = array($orifice_size, $aspect_ratio, $expansion_ratio, $normalized_orifice_length, $normalized_water_inlet, $normalized_oil_inlet, $flow_rate_ratio, $capillary_number);
$constraint_names = array("orifice_size", "aspect_ratio", "expansion_ratio", "normalized_orifice_length",
			"normalized_water_inlet", "normalized_oil_inlet", "flow_rate_ratio", "capillary_number");

$desired_vals = array($generation_rate, $droplet_size);
$desired_vals_names = array("generation_rate","droplet_size");

$DAFD_location = "/home/chris/Work/DAFD/DAFD/";
$file = $DAFD_location . "cmd_inputs.txt";
file_put_contents($file, "CONSTRAINTS\n");

for($i=0; $i<sizeof($constraints); $i++)
{
	$value = $constraints[$i];
	if(!empty($value))
	{
		if (strpos($value, '-') !== false)
		{
			$current = $constraint_names[$i] . "=" . explode("-",$value)[0] . ":" . explode("-",$value)[1] . "\n";
			file_put_contents($file, $current, FILE_APPEND);
		}
		else
		{
			$current = $constraint_names[$i] . "=" . $value . ":" . $value . "\n";
			file_put_contents($file, $current, FILE_APPEND);
		}
	}
}

file_put_contents($file, "DESIRED_VALS\n", FILE_APPEND);

for($i=0; $i<sizeof($desired_vals); $i++)
{
	$value = $desired_vals[$i];
	if(!empty($value))
	{
		$current = $desired_vals_names[$i] . "=" . $value . "\n";
		file_put_contents($file, $current, FILE_APPEND);
	}
}


$outputs = shell_exec($DAFD_location . "venv/bin/python3 " . $DAFD_location . "DAFD_CMD.py");
$arr_outs = explode("|",explode("BEGIN:",$outputs)[1]);
?>

<div>
    <h1>Suggested Parameters</h1>

    Orifice Width (um):
    <?php
    echo $arr_outs[0];
    ?>
    <br>

    Aspect Ratio (Channel Depth Divided By Orifice Width):
    <?php
    echo $arr_outs[1];
    ?>
    <br>

    Expansion Ratio (Outlet Channel Width Divided By Orifice Width):
    <?php
    echo $arr_outs[2];
    ?>
    <br>

    Normalized Orifice Length (Orifice Length Divided By Orifice Width):
    <?php
    echo $arr_outs[3];
    ?>
    <br>

    Normalized Water Inlet Width (Water Inlet Width Divided By Orifice Width):
    <?php
    echo $arr_outs[4];
    ?>
    <br>

    Normalized Oil Inlet Width (Oil Inlet Width Divided By Orifice Width):
    <?php
    echo $arr_outs[5];
    ?>
    <br>

    Flow Rate Ratio (Oil Flow Rate Divided By Water Flow Rate):
    <?php
    echo $arr_outs[6];
    ?>
    <br>

    Capillary Number:
    <?php
    echo $arr_outs[7];
    ?>
    <br>

    Point Source:
    <?php
    echo $arr_outs[8];
    ?>
    <br>

</div>

<br>
<br>

<div>
    <h1>Predicted Outputs</h1>

    Generation Rate (Hz):
    <?php
    echo $arr_outs[9];
    ?>
    <br>

    Droplet Size (um):
    <?php
    echo $arr_outs[10];
    ?>
    <br>

    Regime:
    <?php
    echo $arr_outs[11];
    ?>
    <br>

</div>

<br>
<br>

<div>
    <h1>Calculated Values</h1>


    Oil Flow Rate (ml/hr):
    <?php
    echo $arr_outs[12];
    ?>
    <br>

    Water Flow Rate (ul/min):
    <?php
    echo $arr_outs[13];
    ?>
    <br>

    Droplet Inferred Size (um):
    <?php
    echo $arr_outs[14];
    ?>
    <br>
</div>

</body>
