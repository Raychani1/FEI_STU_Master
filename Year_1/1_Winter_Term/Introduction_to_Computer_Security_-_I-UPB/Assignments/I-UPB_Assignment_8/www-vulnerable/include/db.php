<?php
/**
 */
//Zobrazovanie chyb okrem notice
error_reporting(E_ALL & ~E_NOTICE);
ini_set('display_errors', 1);

$db_host = "localhost";
$db_user = "udpb";
$db_pass = "udpb";
$db_name = "udpb";
$db = new mysqli($db_host, $db_user, $db_pass, $db_name);
if ($db->connect_errno) {
    echo "Failed to connect to MySQL: (" . $db->connect_errno . ") " . $db->connect_error;
}
$db->set_charset("utf8");
