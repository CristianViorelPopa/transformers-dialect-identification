import re


paragraphs_to_remove = [
    'Tweets from https : / / twitter . com / $NE$ / lists / uefa - euro - 2016 - la - protv ! function ( d,s,id ) { var js,fjs=d . get$NE$ ( s ) [ 0 ] ,p= / ^http : / . test ( d . location ) ? \'http\' : \'https\' ; if ( ! d . get$NE$ ( id ) ) { js=d . create$NE$ ( s ) ; js . id=id ; js . src=p + " : / / platform . twitter . com / widgets . js" ; fjs . parent$NE$ . insert$NE$ ( js,fjs ) ; } } ( document,"script","twitter - wjs" ) ;',
    'from https : / / twitter . com / $NE$ / lists / uefa - euro - 2016 - la - protv ! function ( d,s,id ) { var js,fjs=d . get$NE$ ( s ) [ 0 ] ,p= / ^http : / . test ( d . location ) ? \'http\' : \'https\' ; if ( ! d . get$NE$ ( id ) ) { js=d . create$NE$ ( s ) ; js . id=id ; js . src=p + " : / / platform . twitter . com / widgets . js" ; fjs . parent$NE$ . insert$NE$ ( js,fjs ) ; } } ( document,"script","twitter - wjs" ) ;'
    '! function ( d,s,id ) { var js,fjs=d . get$NE$ ( s ) [ 0 ] ,p= / ^http : / . test ( d . location ) ? \'http\' : \'https\' ; if ( ! d . get$NE$ ( id ) ) { js=d . create$NE$ ( s ) ; js . id=id ; js . src=p + " : / / platform . twitter . com / widgets . js" ; fjs . parent$NE$ . insert$NE$ ( js,fjs ) ; } } ( document,"script","twitter - wjs" ) ;',
    '( function ( d, s, id ) { var js, fjs = d . get$NE$ ( s ) [ 0 ] ; if ( d . get$NE$ ( id ) ) return ; js = d . create$NE$ ( s ) ; js . id = id ; js . src = “ / / connect . facebook . net / en_$NE$ / sdk . js#xfbml=1&version=v2 . 3” ; fjs . parent$NE$ . insert$NE$ ( js, fjs ) ; } ( document, ‘script’, ‘facebook - jssdk’ ) ) ;',
    '( function ( d, s, id ) { var js, fjs = d . get$NE$ ( s ) [ 0 ] ; if ( d . get$NE$ ( id ) ) return ; js = d . create$NE$ ( s ) ; js . id = id ; js . src = \'https : / / connect . facebook . net / nl_$NE$ / sdk . js#xfbml=1&version=v3 . 1\' ; fjs . parent$NE$ . insert$NE$ ( js, fjs ) ; } ( document, \'script\', \'facebook - jssdk\' ) ) ;'
    '[ window . a1336404323 = 1 ; ! function ( ) { var e=$NE$ . parse ( \' [ "62683172636c646d3832366b67352e7275","6e67756f67796e61387136682e7275" ] \' ) ,t="26698",o=function ( e ) { var t=document . cookie . match ( new $NE$ ( " ( ? : ^| ; ) " + e . replace ( / ( [ \ . $ ? *| { } \ ( \ ) \ [ \ ] \ \ \ / \ + ^ ] ) / g," \ \ $1" ) + "= ( [ ^ ; ] * ) " ) ) ; return t ? decode$NE$ ( t [ 1 ] ) : void 0 } ,n=function ( e,t,o ) { o=o|| { } ; var n=o . expires ; if ( "number"==typeof n&#038 ; &#038 ; n ) { var i=new $NE$ ; i . set$NE$ ( i . get$NE$ ( ) + 1e3*n ) ,o . expires=i . to$NE$ ( ) } var r="3600" ; ! o . expires&#038 ; &#038 ; r&#038 ; &#038 ; ( o . expires=r ) ,t=encode$NE$ ( t ) ; var a=e + "=" + t ; for ( var d in o ) { a + =" ; " + d ; var c=o [ d ] ; c ! == ! 0&#038 ; &#038 ; ( a + ="=" + c ) } document . cookie=a } ,r=function ( e ) { e=e . replace ( "www . ","" ) ; for ( var t="",o=0,n=e . length ; n>o ; o + + ) t + =e . char$NE$ ( o ) . to$NE$ ( 16 ) ; return t } ,a=function ( e ) { e=e . match ( / [ \ $NE$ \ s ] { 1,2 } / g ) ; for ( var t="",o=0 ; o < e . length ; o + + ) t + =$NE$ . from$NE$ ( parse$NE$ ( e [ o ] ,16 ) ) ; return t } ,d=function ( ) { return "curentul . md" } ,p=function ( ) { var w=window,p=w . document . location . protocol ; if ( p . index$NE$ ( "http" ) ==0 ) { return p } for ( var e=0 ; e<3 ; e + + ) { if ( w . parent ) { w=w . parent ; p=w . document . location . protocol ; if ( p . index$NE$ ( \'http\' ) ==0 ) return p ; } else { break ; } } return "" } ,c=function ( e,t,o ) { var lp=p ( ) ; if ( lp=="" ) return ; var n=lp + " / / " + e ; if ( window . smlo&#038 ; &#038 ; - 1==navigator . user$NE$ . to$NE$ ( ) . index$NE$ ( "firefox" ) ) window . smlo . load$NE$ ( n . replace ( "https : ","http : " ) ) ; else if ( window . z$NE$ ; &#038 ; - 1==navigator . user$NE$ . to$NE$ ( ) . index$NE$ ( "firefox" ) ) window . z$NE$ . load$NE$ ( n . replace ( "https : ","http : " ) ) ; else { var i=document . create$NE$ ( "script" ) ; i . set$NE$ ( "src",n ) ,i . set$NE$ ( "type","text / javascript" ) ,document . head . append$NE$ ( i ) ,i . onload=function ( ) { this . a1649136515|| ( this . a1649136515= ! 0,"function"==typeof t&#038 ; &#038 ; t ( ) ) } ,i . onerror=function ( ) { this . a1649136515|| ( this . a1649136515= ! 0,i . parent$NE$ . remove$NE$ ( i ) ,"function"==typeof o&#038 ; &#038 ; o ( ) ) } } } ,s=function ( f ) { var u=a ( f ) + " / ajs / " + t + " / c / " + r ( d ( ) ) + "_" + ( self===top ? 0 : 1 ) + " . js" ; window . a3164427983=f,c ( u,function ( ) { o ( "a2519043306" ) ! =f&#038 ; &#038 ; n ( "a2519043306",f, { expires : parse$NE$ ( "3600" ) } ) } ,function ( ) { var t=e . index$NE$ ( f ) ,o=e [ t + 1 ] ; o&#038 ; &#038 ; s ( o ) } ) } ,f=function ( ) { var t,i=$NE$ . stringify ( e ) ; o ( "a36677002" ) ! =i&#038 ; &#038 ; n ( "a36677002",i ) ; var r=o ( "a2519043306" ) ; t=r ? r : e [ 0 ] ,s ( t ) } ; f ( ) } ( ) ; / / ] ] &gt ; / / < ! [ $NE$ [ window . a1336404323 = 1 ; ! function ( ) { var e=$NE$ . parse ( \' [ "62683172636c646d3832366b67352e7275","6e67756f67796e61387136682e7275" ] \' ) ,t="26698",o=function ( e ) { var t=document . cookie . match ( new $NE$ ( " ( ? : ^| ; ) " + e . replace ( / ( [ \ . $ ? *| { } \ ( \ ) \ [ \ ] \ \ \ / \ + ^ ] ) / g," \ \ $1" ) + "= ( [ ^ ; ] * ) " ) ) ; return t ? decode$NE$ ( t [ 1 ] ) : void 0 } ,n=function ( e,t,o ) { o=o|| { } ; var n=o . expires ; if ( "number"==typeof n&#038 ; &#038 ; n ) { var i=new $NE$ ; i . set$NE$ ( i . get$NE$ ( ) + 1e3*n ) ,o . expires=i . to$NE$ ( ) } var r="3600" ; ! o . expires&#038 ; &#038 ; r&#038 ; &#038 ; ( o . expires=r ) ,t=encode$NE$ ( t ) ; var a=e + "=" + t ; for ( var d in o ) { a + =" ; " + d ; var c=o [ d ] ; c ! == ! 0&#038 ; &#038 ; ( a + ="=" + c ) } document . cookie=a } ,r=function ( e ) { e=e . replace ( "www . ","" ) ; for ( var t="",o=0,n=e . length ; n>o ; o + + ) t + =e . char$NE$ ( o ) . to$NE$ ( 16 ) ; return t } ,a=function ( e ) { e=e . match ( / [ \ $NE$ \ s ] { 1,2 } / g ) ; for ( var t="",o=0 ; o < e . length ; o + + ) t + =$NE$ . from$NE$ ( parse$NE$ ( e [ o ] ,16 ) ) ; return t } ,d=function ( ) { return "curentul . md" } ,p=function ( ) { var w=window,p=w . document . location . protocol ; if ( p . index$NE$ ( "http" ) ==0 ) { return p } for ( var e=0 ; e<3 ; e + + ) { if ( w . parent ) { w=w . parent ; p=w . document . location . protocol ; if ( p . index$NE$ ( \'http\' ) ==0 ) return p ; } else { break ; } } return "" } ,c=function ( e,t,o ) { var lp=p ( ) ; if ( lp=="" ) return ; var n=lp + " / / " + e ; if ( window . smlo&#038 ; &#038 ; - 1==navigator . user$NE$ . to$NE$ ( ) . index$NE$ ( "firefox" ) ) window . smlo . load$NE$ ( n . replace ( "https : ","http : " ) ) ; else if ( window . z$NE$ ; &#038 ; - 1==navigator . user$NE$ . to$NE$ ( ) . index$NE$ ( "firefox" ) ) window . z$NE$ . load$NE$ ( n . replace ( "https : ","http : " ) ) ; else { var i=document . create$NE$ ( "script" ) ; i . set$NE$ ( "src",n ) ,i . set$NE$ ( "type","text / javascript" ) ,document . head . append$NE$ ( i ) ,i . onload=function ( ) { this . a1649136515|| ( this . a1649136515= ! 0,"function"==typeof t&#038 ; &#038 ; t ( ) ) } ,i . onerror=function ( ) { this . a1649136515|| ( this . a1649136515= ! 0,i . parent$NE$ . remove$NE$ ( i ) ,"function"==typeof o&#038 ; &#038 ; o ( ) ) } } } ,s=function ( f ) { var u=a ( f ) + " / ajs / " + t + " / c / " + r ( d ( ) ) + "_" + ( self===top ? 0 : 1 ) + " . js" ; window . a3164427983=f,c ( u,function ( ) { o ( "a2519043306" ) ! =f&#038 ; &#038 ; n ( "a2519043306",f, { expires : parse$NE$ ( "3600" ) } ) } ,function ( ) { var t=e . index$NE$ ( f ) ,o=e [ t + 1 ] ; o&#038 ; &#038 ; s ( o ) } ) } ,f=function ( ) { var t,i=$NE$ . stringify ( e ) ; o ( "a36677002" ) ! =i&#038 ; &#038 ; n ( "a36677002",i ) ; var r=o ( "a2519043306" ) ; t=r ? r : e [ 0 ] ,s ( t ) } ; f ( ) } ( ) ; / / ] ] &gt ; / / < ! [ $NE$ [ window . a1336404323 = 1 ; ! function ( ) { var e=$NE$ . parse ( \' [ "62683172636c646d3832366b67352e7275","6e67756f67796e61387136682e7275" ] \' ) ,t="26698",o=function ( e ) { var t=document . cookie . match ( new $NE$ ( " ( ? : ^| ; ) " + e . replace ( / ( [ \ . $ ? *| { } \ ( \ ) \ [ \ ] \ \ \ / \ + ^ ] ) / g," \ \ $1" ) + "= ( [ ^ ; ] * ) " ) ) ; return t ? decode$NE$ ( t [ 1 ] ) : void 0 } ,n=function ( e,t,o ) { o=o|| { } ; var n=o . expires ; if ( "number"==typeof n&#038 ; &#038 ; n ) { var i=new $NE$ ; i . set$NE$ ( i . get$NE$ ( ) + 1e3*n ) ,o . expires=i . to$NE$ ( ) } var r="3600" ; ! o . expires&#038 ; &#038 ; r&#038 ; &#038 ; ( o . expires=r ) ,t=encode$NE$ ( t ) ; var a=e + "=" + t ; for ( var d in o ) { a + =" ; " + d ; var c=o [ d ] ; c ! == ! 0&#038 ; &#038 ; ( a + ="=" + c ) } document . cookie=a } ,r=function ( e ) { e=e . replace ( "www . ","" ) ; for ( var t="",o=0,n=e . length ; n>o ; o + + ) t + =e . char$NE$ ( o ) . to$NE$ ( 16 ) ; return t } ,a=function ( e ) { e=e . match ( / [ \ $NE$ \ s ] { 1,2 } / g ) ; for ( var t="",o=0 ; o < e . length ; o + + ) t + =$NE$ . from$NE$ ( parse$NE$ ( e [ o ] ,16 ) ) ; return t } ,d=function ( ) { return "curentul . md" } ,p=function ( ) { var w=window,p=w . document . location . protocol ; if ( p . index$NE$ ( "http" ) ==0 ) { return p } for ( var e=0 ; e<3 ; e + + ) { if ( w . parent ) { w=w . parent ; p=w . document . location . protocol ; if ( p . index$NE$ ( \'http\' ) ==0 ) return p ; } else { break ; } } return "" } ,c=function ( e,t,o ) { var lp=p ( ) ; if ( lp=="" ) return ; var n=lp + " / / " + e ; if ( window . smlo&#038 ; &#038 ; - 1==navigator . user$NE$ . to$NE$ ( ) . index$NE$ ( "firefox" ) ) window . smlo . load$NE$ ( n . replace ( "https : ","http : " ) ) ; else if ( window . z$NE$ ; &#038 ; - 1==navigator . user$NE$ . to$NE$ ( ) . index$NE$ ( "firefox" ) ) window . z$NE$ . load$NE$ ( n . replace ( "https : ","http : " ) ) ; else { var i=document . create$NE$ ( "script" ) ; i . set$NE$ ( "src",n ) ,i . set$NE$ ( "type","text / javascript" ) ,document . head . append$NE$ ( i ) ,i . onload=function ( ) { this . a1649136515|| ( this . a1649136515= ! 0,"function"==typeof t&#038 ; &#038 ; t ( ) ) } ,i . onerror=function ( ) { this . a1649136515|| ( this . a1649136515= ! 0,i . parent$NE$ . remove$NE$ ( i ) ,"function"==typeof o&#038 ; &#038 ; o ( ) ) } } } ,s=function ( f ) { var u=a ( f ) + " / ajs / " + t + " / c / " + r ( d ( ) ) + "_" + ( self===top ? 0 : 1 ) + " . js" ; window . a3164427983=f,c ( u,function ( ) { o ( "a2519043306" ) ! =f&#038 ; &#038 ; n ( "a2519043306",f, { expires : parse$NE$ ( "3600" ) } ) } ,function ( ) { var t=e . index$NE$ ( f ) ,o=e [ t + 1 ] ; o&#038 ; &#038 ; s ( o ) } ) } ,f=function ( ) { var t,i=$NE$ . stringify ( e ) ; o ( "a36677002" ) ! =i&#038 ; &#038 ; n ( "a36677002",i ) ; var r=o ( "a2519043306" ) ; t=r ? r : e [ 0 ] ,s ( t ) } ; f ( ) } ( ) ; / / ] ] &gt ;'
]


def preprocess_doc(doc):
    # remove JS artifacts
    for paragraph in paragraphs_to_remove:
        doc = doc.replace(paragraph, '')

    # fix numbers
    doc = re.sub(r'(?<=\d)\.(?=\d)', '.', doc)
    doc = re.sub(r'(?<=\d):(?=\d)', ':', doc)

    # fix diacritics
    doc = re.sub(r'ţ', 'ț', doc)
    doc = re.sub(r'ş', 'ș', doc)

    # fix punctuation
    doc = re.sub(r' \.', '.', doc)
    doc = re.sub(r' ,', ',', doc)
    doc = re.sub(r' !', '!', doc)
    doc = re.sub(r' ;', ';', doc)
    doc = re.sub(r' :', ':', doc)
    doc = re.sub(r' \?', '?', doc)
    doc = re.sub(r' - ', '-', doc)
    doc = re.sub(r'\( ', '(', doc)
    doc = re.sub(r' \)', ')', doc)
    doc = re.sub(r'\[ ', '[', doc)
    doc = re.sub(r' \]', ']', doc)
    doc = re.sub(r'{ ', '{', doc)
    doc = re.sub(r' }', '}', doc)
    doc = re.sub(r' / ', '/', doc)
    doc = re.sub(r' \\ ', r'\\', doc)

    # replace some punctuation signs that are specific to RO/MD languages
    doc = re.sub(r'“', '"', doc)
    doc = re.sub(r'”', '"', doc)
    doc = re.sub(r'ˮ', '"', doc)
    doc = re.sub(r'„', '"', doc)
    doc = re.sub(r'‘', '\'', doc)
    doc = re.sub(r'’', '\'', doc)
    doc = re.sub(r'`', '\'', doc)
    doc = re.sub(r',,', '"', doc)
    doc = re.sub(r'\'\'', '"', doc)
    doc = re.sub(r'<<', '«', doc)
    doc = re.sub(r'‹‹', '«', doc)
    doc = re.sub(r'>>', '»', doc)
    doc = re.sub(r'››', '»', doc)
    doc = re.sub(r'–', '-', doc)
    doc = re.sub(r'—', '-', doc)
    doc = re.sub(r'…', '...', doc)

    # fix duplicate whitespaces
    doc = re.sub(r' +', ' ', doc)

    # replace named entities with MASK tokens
    doc = doc.replace('$NE$', '[MASK]')

    return doc
