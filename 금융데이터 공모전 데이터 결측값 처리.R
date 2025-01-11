library(readxl)
df=read_excel("finaldata (3).xlsx")
df
sum(is.na(df))
library(visdat)
vis_miss(df)
#install.packages("VIM")
library(VIM)
str(df)
simple=df[c("a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","aaa","aa")]
#install.packages("MICE")
library(mice)
set.seed(123)
simple_imp <- complete(mice(simple))

df$a=simple_imp$a
df$b=simple_imp$b
df$c=simple_imp$c
df$d=simple_imp$d
df$e=simple_imp$e
df$f=simple_imp$f
df$g=simple_imp$g
df$h=simple_imp$h
df$i=simple_imp$i
df$j=simple_imp$j
df$k=simple_imp$k
df$l=simple_imp$l
df$m=simple_imp$m
df$n=simple_imp$n
df$o=simple_imp$o
df$p=simple_imp$p
df$q=simple_imp$q
df$r=simple_imp$r
df$s=simple_imp$s
df$aaa=simple_imp$aaa
df$aa=simple_imp$aa

str(df)
sum(is.na(df))
vis_miss(df)
#write.csv(df,file="결측치대체2.csv")

