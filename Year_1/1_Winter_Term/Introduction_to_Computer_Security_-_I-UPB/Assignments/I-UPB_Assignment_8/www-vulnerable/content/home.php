<?php
include_once(__DIR__.DIRECTORY_SEPARATOR.'../include/db.php');
?>
<!-- Post -->

<?php
// A7 - vytvor .htaccess v priecinku content a nastav deny from all
function replace_n($str){
$order   = array('\n');
$replace = '<br>';
return $newstr = str_replace($order, $replace, $str);
}

function replace_br($str){
$order   = array("<br>");
$replace = '\n';
return $newstr = str_replace($order, $replace, $str);
}


if(@$_GET['id']<=0) {
    //time based blind sqli maybe ? ;) 
    $articles = $db->query("SELECT * FROM articles WHERE status=1 ORDER BY id DESC");
    while ($article = $articles->fetch_array(MYSQL_ASSOC)) {
        echo '<article class="box post post-excerpt">
        <header>
            <h2><a href="./index.php?id='.$article['id'].'">' . $article['title'] . '</a></h2>
        </header>
        <div class="info">
            <span class="date"><span class="month">' . date("M", strtotime($article['date'])) . '</span> <span class="day">' . date("d", strtotime($article['date'])) . '</span><span class="year">, ' . date("y", strtotime($article['date'])) . '</span></span>
        </div>
        <a href="#" class="image featured"><img src="images/pic01.jpg" alt="" /></a>';
        $replaced_placeholder = replace_br($article['content']);
        $new = htmlspecialchars($replaced_placeholder,ENT_QUOTES);
	$replaced_n = replace_n($new);
        echo $replaced_n;
     echo '</article>';
    }
}else{
    $articles = $db->query("SELECT * FROM articles WHERE id='".$db->real_escape_string($_GET['id'])."'");
    $article = $articles->fetch_array(MYSQL_ASSOC);
        echo '<article class="box post post-excerpt">
        <header>
            <h2><a href="#">' . $article['title'] . '</a></h2>
        </header>
        <div class="info">
            <span class="date"><span class="month">' . date("M", strtotime($article['date'])) . '</span> <span class="day">' . date("d", strtotime($article['date'])) . '</span><span class="year">, ' . date("y", strtotime($article['date'])) . '</span></span>
        </div>
        <a href="#" class="image featured"><img src="images/pic01.jpg" alt="" /></a>';
        $replaced_placeholder = replace_br($article['content']);
        $new = htmlspecialchars($replaced_placeholder,ENT_QUOTES);
        $replaced_n = replace_n($new);
        echo $replaced_n;
   echo  '</article>';
}
?>
