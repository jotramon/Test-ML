-- Only the columns DelayModel needs: features (OPERA, TIPOVUELO, MES) and the
-- schedule columns the delay target is derived from.
SELECT
  Fecha_I,
  Fecha_O,
  OPERA,
  TIPOVUELO,
  MES
FROM `{project}.{dataset}.{raw_table}`
