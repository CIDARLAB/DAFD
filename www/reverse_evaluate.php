
<?php

$orifice_size=$_POST['orifice-size'];
$aspect_ratio=$_POST['aspect-ratio'];
$expansion_ratio=$_POST['expansion-ratio'];
$normalized_orifice_length=$_POST['normalized-orifice-length'];
$normalized_water_inlet=$_POST['normalized-water-inlet'];
$normalized_oil_inlet=$_POST['normalized-oil-inlet'];
$flow_rate_ratio=$_POST['flow-rate-ratio'];
$capillary_number=$_POST['capillary-number'];
$regime=$_POST['regime'];

$generation_rate=$_POST['generation-rate'];
$droplet_size=$_POST['droplet-size'];

$constraints = array($orifice_size, $aspect_ratio, $expansion_ratio, $normalized_orifice_length, $normalized_water_inlet, $normalized_oil_inlet, $flow_rate_ratio, $capillary_number, $regime);
$constraint_names = array("orifice_size", "aspect_ratio", "expansion_ratio", "normalized_orifice_length",
			"normalized_water_inlet", "normalized_oil_inlet", "flow_rate_ratio", "capillary_number","regime");

$desired_vals = array($generation_rate, $droplet_size);
$desired_vals_names = array("generation_rate","droplet_size");

$DAFD_location = "/home/dafdadmin/DAFD/";
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

<div style="text-align: center">
<div class="div_float">
    <h1>Optimal Design Parameters</h1>

    <h2>Geometric Parameters</h2>
    Orifice Width (um):
    <?php
    echo round(floatval($arr_outs[0]), 2);
    ?>
    <br>

    Channel Depth (um):
    <?php
    echo round(floatval($arr_outs[1]) * floatval($arr_outs[0]), 2);
    ?>
    <br>

    Outlet Channel Width (um):
    <?php
    echo round(floatval($arr_outs[2]) * floatval($arr_outs[0]), 2);
    ?>
    <br>

    Orifice Length (um):
    <?php
    echo round(floatval($arr_outs[3]) * floatval($arr_outs[0]), 2);
    ?>
    <br>

    Water Inlet Width (um):
    <?php
    echo round(floatval($arr_outs[4]) * floatval($arr_outs[0]), 2);
    ?>
    <br>

    Oil Inlet Width (um):
    <?php
    echo round(floatval($arr_outs[5]) * floatval($arr_outs[0]), 2);
    ?>
    <br>

    <h2>Flow Conditions</h2>
    Flow Rate Ratio (Oil Flow Rate Divided By Water Flow Rate):
    <?php
    echo round(floatval($arr_outs[6]), 2);
    ?>
    <br>

    Capillary Number:
    <?php
    echo round(floatval($arr_outs[7]), 3);
    ?>
    <br>

    <h2>Optimization Strategy</h2>
    Point Source:
    <?php
    echo $arr_outs[8];
    ?>
    <br>

</div>

<br>
<br>

<div class="div_float">
    <h1>Predicted Performance</h1>

    Generation Rate (Hz):
    <?php
    echo round(floatval($arr_outs[9]), 3);
    ?>
    <br>

    Droplet Diameter (um):
    <?php
    echo round(floatval($arr_outs[10]), 3);
    ?>
    <br>

    Inferred Droplet Diameter (um):
    <?php
    echo round(floatval($arr_outs[14]), 3);
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

<div class="div_float">
    <h1>Flow Conditions</h1>


    Oil Flow Rate (ml/hr):
    <?php
    echo round(floatval($arr_outs[12]), 3);
    ?>
    <br>

    Water Flow Rate (ul/min):
    <?php
    echo round(floatval($arr_outs[13]), 3);
    ?>
    <br>
</div>

<br>
<br>

<div class="div_float">
    <h1>Single Cell Encapsulation</h1>
    Lambda (ratio of cells to droplets): <input type="text" id="lambda" value="0.1" onchange="calculateConc()" onload="calculateConc()"><br>
    <input type="text" id="water_flow_rate" value="<?php echo round(floatval($arr_outs[13]), 3);?>" hidden>
    <input type="text" id="generation_rate" value="<?php echo round(floatval($arr_outs[9]), 3);?>" hidden>
    Cell concentration (cells per ul) : <label type="text" id="cellconc" >1</label><br>
    <br>
</div>

</div>

