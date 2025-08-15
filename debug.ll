define void @"fused_6aaf920e"({i8*, i8*, i64, i64, [8 x i64], [8 x i64]}* %".1", {i8*, i8*, i64, i64, [8 x i64], [8 x i64]}* %".2", {i8*, i8*, i64, i64, [8 x i64], [8 x i64]}* %".3")
{
entry:
  br label %"loop"
loop:
  %".6" = phi  i64 [0, %"entry"], [%".23", %"loop"]
  %".7" = getelementptr {i8*, i8*, i64, i64, [8 x i64], [8 x i64]}, {i8*, i8*, i64, i64, [8 x i64], [8 x i64]}* %".1", i32 0, i32 1
  %".8" = load i8*, i8** %".7"
  %".9" = bitcast i8* %".8" to double*
  %".10" = getelementptr double, double* %".9", i64 %".6"
  %".11" = load double, double* %".10"
  %".12" = getelementptr {i8*, i8*, i64, i64, [8 x i64], [8 x i64]}, {i8*, i8*, i64, i64, [8 x i64], [8 x i64]}* %".2", i32 0, i32 1
  %".13" = load i8*, i8** %".12"
  %".14" = bitcast i8* %".13" to double*
  %".15" = getelementptr double, double* %".14", i64 %".6"
  %".16" = load double, double* %".15"
  %".17" = fmul double %".11", %".16"
  %".18" = getelementptr {i8*, i8*, i64, i64, [8 x i64], [8 x i64]}, {i8*, i8*, i64, i64, [8 x i64], [8 x i64]}* %".3", i32 0, i32 1
  %".19" = load i8*, i8** %".18"
  %".20" = bitcast i8* %".19" to double*
  %".21" = getelementptr double, double* %".20", i64 %".6"
  store double %".17", double* %".21"
  %".23" = add i64 %".6", 1
  %".24" = icmp slt i64 %".23", 4
  br i1 %".24", label %"loop", label %"done"
done:
  ret void
}
